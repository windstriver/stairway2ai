import torch
import numpy as np

"""
# Tensors

- data structure similar to arrays and matrices (similar API with NumPy ndarray)
- to encode inputs, outputs, and parameters of models
- can run on GPUs to accelerate computing

"""

# ## Tensor Initialization
# ### Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data) # data type inferred
print(x_data.dtype) # => torch.int64

# ### From a Numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array) # bridge with NumPy

# ### From another tensor
x_ones = torch.ones_like(x_data) # retains the properties (shape, dtype) of x_data
print(f"Ones Tensor: \n {x_ones} with dtype of {x_ones.dtype} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the dtype of x_data
print(f"Random Tensor: \n {x_rand} with dtype of {x_rand.dtype} \n")

# ### With random or constant values
shape = (2, 3,) # tuple of tensor dimensions
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# ## Tensor Attributes
# Tensor attributes describe their shape, datatype, and the device on which they are stored
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# ## Tensor Operations
# - over 100 tensor operations (transposing, indexing, slicing, math operations, linear algebra, random sampling, and more)
# - can be run on GPU
# check if GPU is available, if yes, move tensor to it
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")

# ### Standard numpy-like indexing and slicing
# ### Joining tensors: torch.cat, torch.stack
# ### Multiplying tensors
# - this computes the element-wise product
print(f"Multiplying with element-wise product: tensor.mul(tensor) \n {tensor.mul(tensor)}")
# - alternative syntax
print(f"Multiplying with element-wise product: tensor * tensor \n {tensor * tensor}")

# - this computes the matrix multiplication between two tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)}")
# - alternative syntax
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# - in-place operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# ## Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations
# changing one will change the other.
