.. Create and Add Two Tensors Example:

Basic Operations with TT-NN
###########################


Lets create the example file,
``ttnn_basic_operations.py``

Import the necessary libraries
------------------------------

.. code-block:: python

   import torch
   import numpy as np
   import ttnn

Open Tenstorrent device
-----------------------

Create necessary device on which we will run our program.

.. code-block:: python

   # Open Tenstorrent device
   device = ttnn.open_device(device_id=0)


Helper Functions for Tensor Preparation
---------------------------------------

Lets create a few helper functions for convering from PyTorch tensors to TT-NN tiled tensors, and initializing some host-side tensors.

.. code-block:: python

   # Helper to create a TT-NN tensor from torch with TILE_LAYOUT and bfloat16
   def to_tt_tile(torch_tensor):
      return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

   # Helper to create (32, 32) torch tensor from scalar or numpy
   def create_host_tensor(fill_value):
      if isinstance(fill_value, (int, float)):
         return torch.full((32, 32), fill_value, dtype=torch.float32)
      elif isinstance(fill_value, np.ndarray):
         return torch.from_numpy(fill_value.astype(np.float32))
      else:
         raise ValueError("Unsupported type for fill_value")

Host Tensor Creation
--------------------

Create several tensors for our tests and fill with different values. We will use these tensors to demonstrate various operations.

.. code-block:: python

   print("\n--- TT-NN Tensor Creation with Tiles (32x32) ---")
   host_t1 = create_host_tensor(1)
   host_t2 = torch.zeros((32, 32), dtype=torch.float32)
   host_t3 = torch.ones((32, 32), dtype=torch.float32)
   host_t4 = torch.rand((32, 32), dtype=torch.float32)
   host_np_array = np.array([[5, 6], [7, 8]]).repeat(16, axis=0).repeat(16, axis=1)
   host_t5 = create_host_tensor(host_np_array)

Convert Host Tensors to TT-NN Tiled Tensors
-------------------------------------------

Tensix cores operate most efficiently on tiled data, allowing them to perform a large amount of compute in parallel.

.. code-block:: python

   tt_t1 = to_tt_tile(host_t1)
   tt_t2 = to_tt_tile(host_t2)
   tt_t3 = to_tt_tile(host_t3)
   tt_t4 = to_tt_tile(host_t4)
   tt_t5 = to_tt_tile(host_t5)

Tile-Based Arithmetic Operations
--------------------------------

Lets use some of the tensors we created and perform different operations on them.

.. code-block:: python

   print("\n--- TT-NN Tensor Operations on (32x32) Tiles ---")
   add_result = ttnn.add(tt_t3, tt_t4)
   mul_result = ttnn.mul(tt_t4, tt_t5)
   matmul_result = ttnn.matmul(tt_t3, tt_t4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

Simulated Broadcasting (Row Vector Expansion)
---------------------------------------------

Lets simulated broadcasting a row vector across a tile. This is useful for operations that require expanding a smaller tensor to match the dimensions of a larger one.

.. code-block:: python

   print("\n--- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---")
   broadcast_vector = torch.tensor([[1.0] * 32], dtype=torch.float32).repeat(32, 1)
   broadcast_tt = to_tt_tile(broadcast_vector)
   broadcast_add_result = ttnn.add(tt_t4, broadcast_tt)


ull example and output
-----------------------

Lets put everything together in a complete example that can be run directly. This example will open a Tenstorrent device, create some input tensors and perform operations on them, print the output tensors, and close the device.

.. literalinclude:: ttnn_tutorials_basic_python/ttnn_basic_operations.py
   :caption: Source Code

Running this script will output the operation results as shown below

.. code-block:: console

   $ python3 $TT_METAL_HOME/ttnn/tutorials/basic_python/ttnn_basic_operations.py
   --- TT-NN Tensor Creation with Tiles (32x32) ---
   Tensor from fill value 1:
      tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        ...,
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.]], dtype=torch.bfloat16)
   Zeros:
      tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.bfloat16)
   Ones:
      tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        ...,
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.]], dtype=torch.bfloat16)
   Random:
      tensor([[0.5117, 0.2598, 0.2354,  ..., 0.2578, 0.2676, 0.7773],
        [0.1602, 0.4082, 0.4453,  ..., 0.7500, 0.0840, 0.7266],
        [0.7461, 0.0540, 0.6367,  ..., 0.1216, 0.2275, 0.4883],
        ...,
        [0.9180, 0.3691, 0.9492,  ..., 0.6484, 0.1206, 0.7891],
        [0.8516, 0.0972, 0.3594,  ..., 0.6094, 0.8164, 0.5195],
        [0.2715, 0.1660, 0.9922,  ..., 0.5547, 0.4023, 0.0664]],
       dtype=torch.bfloat16)
   From expanded NumPy (TT-NN):
      tensor([[5., 5., 5.,  ..., 6., 6., 6.],
        [5., 5., 5.,  ..., 6., 6., 6.],
        [5., 5., 5.,  ..., 6., 6., 6.],
        ...,
        [7., 7., 7.,  ..., 8., 8., 8.],
        [7., 7., 7.,  ..., 8., 8., 8.],
        [7., 7., 7.,  ..., 8., 8., 8.]], dtype=torch.bfloat16)

   --- TT-NN Tensor Operations on (32x32) Tiles ---
   Addition:
      tensor([[1.5156, 1.2578, 1.2344,  ..., 1.2578, 1.2656, 1.7812],
        [1.1641, 1.4062, 1.4453,  ..., 1.7500, 1.0859, 1.7266],
        [1.7500, 1.0547, 1.6406,  ..., 1.1250, 1.2266, 1.4922],
        ...,
        [1.9219, 1.3672, 1.9531,  ..., 1.6484, 1.1250, 1.7891],
        [1.8516, 1.1016, 1.3594,  ..., 1.6094, 1.8203, 1.5234],
        [1.2734, 1.1641, 1.9922,  ..., 1.5547, 1.4062, 1.0703]],
       dtype=torch.bfloat16)
   Element-wise Multiplication:
      tensor([[2.5625, 1.2969, 1.1797,  ..., 1.5469, 1.6094, 4.6562],
        [0.8008, 2.0469, 2.2344,  ..., 4.5000, 0.5039, 4.3750],
        [3.7344, 0.2695, 3.1875,  ..., 0.7305, 1.3672, 2.9375],
        ...,
        [6.4375, 2.5781, 6.6562,  ..., 5.1875, 0.9648, 6.3125],
        [5.9688, 0.6797, 2.5156,  ..., 4.8750, 6.5312, 4.1562],
        [1.8984, 1.1641, 6.9375,  ..., 4.4375, 3.2188, 0.5312]],
       dtype=torch.bfloat16)
   Matrix Multiplication:
      tensor([[17.5000, 14.0000, 16.8750,  ..., 16.2500, 16.5000, 18.0000],
        [17.5000, 14.0000, 16.8750,  ..., 16.2500, 16.5000, 18.0000],
        [17.5000, 14.0000, 16.8750,  ..., 16.2500, 16.5000, 18.0000],
        ...,
        [17.5000, 14.0000, 16.8750,  ..., 16.2500, 16.5000, 18.0000],
        [17.5000, 14.0000, 16.8750,  ..., 16.2500, 16.5000, 18.0000],
        [17.5000, 14.0000, 16.8750,  ..., 16.2500, 16.5000, 18.0000]],
       dtype=torch.bfloat16)

   --- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---
   Broadcast Add Result (TT-NN):
      tensor([[1.5156, 1.2578, 1.2344,  ..., 1.2578, 1.2656, 1.7812],
        [1.1641, 1.4062, 1.4453,  ..., 1.7500, 1.0859, 1.7266],
        [1.7500, 1.0547, 1.6406,  ..., 1.1250, 1.2266, 1.4922],
        ...,
        [1.9219, 1.3672, 1.9531,  ..., 1.6484, 1.1250, 1.7891],
        [1.8516, 1.1016, 1.3594,  ..., 1.6094, 1.8203, 1.5234],
        [1.2734, 1.1641, 1.9922,  ..., 1.5547, 1.4062, 1.0703]],
       dtype=torch.bfloat16)
