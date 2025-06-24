.. Basic Operations with TT-NN:

Basic Operations with TT-NN
###########################

We will review a simple example that demonstrates how to create various tensors and
perform basic arithmetic operations on them using TT-NN, a high-level Python API.  These
operations include addition, multiplication, and matrix multiplication, as well as simulating
broadcasting of a row vector across a tile.

Lets create the example file,
``ttnn_basic_operations.py``

Import the necessary libraries
------------------------------

.. code-block:: python

   import torch
   import numpy as np
   import ttnn
   from loguru import logger

Open Tenstorrent device
-----------------------

Create necessary device on which we will run our program.

.. code-block:: python

   # Open Tenstorrent device
   device = ttnn.open_device(device_id=0)


Helper Function for Tensor Preparation
--------------------------------------

Lets create a helper function for convering from PyTorch tensors to TT-NN tiled tensors.

.. code-block:: python

   # Helper to create a TT-NN tensor from torch with TILE_LAYOUT and bfloat16
   def to_tt_tile(torch_tensor):
      return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

Host Tensor Creation
--------------------

Create a tensor for our tests and fill with different values. We will use this and other tensors to demonstrate various operations.

.. code-block:: python

   logger.info("\n--- TT-NN Tensor Creation with Tiles (32x32) ---")
   host_rand = torch.rand((32, 32), dtype=torch.float32)

Convert Host Tensors to TT-NN Tiled Tensors or Create Natively on Device
------------------------------------------------------------------------

Tensix cores operate most efficiently on tiled data, allowing them to perform a large amount of compute in parallel. Where necesasry, lets convert
host tensors to TT-NN tiled tensors using the helper function we created earlier, and transfer them to the TT device.  Alternatively, we can create tensors
natively using TT-NN's tensor creation functions, and initialize them directly on the TT device.  TT-NN calls that create tensors natively on the device are a
more efficient way to create tensors, as they avoid the overhead of transferring data from the host to the device.

.. code-block:: python

   tt_t1 = ttnn.full(
      shape=(32, 32),
      fill_value=1.0,
      dtype=ttnn.float32,
      layout=ttnn.TILE_LAYOUT,
      device=device,
   )

   tt_t2 = ttnn.zeros(
      shape=(32, 32),
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=device,
   )
   tt_t3 = ttnn.ones(
      shape=(32, 32),
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=device,
   )
   tt_t4 = to_tt_tile(host_rand)

   t5 = np.array([[5, 6], [7, 8]], dtype=np.float32).repeat(16, axis=0).repeat(16, axis=1)
   tt_t5 = ttnn.Tensor(t5, device=device, layout=ttnn.TILE_LAYOUT)

Tile-Based Arithmetic Operations
--------------------------------

Lets use some of the tensors we created and perform different operations on them.

.. code-block:: python

   logger.info("\n--- TT-NN Tensor Operations on (32x32) Tiles ---")
   add_result = ttnn.add(tt_t3, tt_t4)
   mul_result = ttnn.mul(tt_t4, tt_t5)
   matmul_result = ttnn.matmul(tt_t3, tt_t4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

Simulated Broadcasting (Row Vector Expansion)
---------------------------------------------

Lets simulated broadcasting a row vector across a tile. This is useful for operations that require expanding a smaller tensor to match the dimensions of a larger one.

.. code-block:: python

   logger.info("\n--- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---")
   broadcast_vector = torch.tensor([[1.0] * 32], dtype=torch.float32).repeat(32, 1)
   broadcast_tt = to_tt_tile(broadcast_vector)
   broadcast_add_result = ttnn.add(tt_t4, broadcast_tt)


Full example and output
-----------------------

Lets put everything together in a complete example that can be run directly. This example will open a Tenstorrent device, create some input tensors and perform operations on them, log the output tensors, and close the device.

.. literalinclude:: ttnn_tutorials_basic_python/ttnn_basic_operations.py
   :caption: Source Code

Running this script will output the operation results as shown below

.. code-block:: console

   $ python3 $TT_METAL_HOME/ttnn/tutorials/basic_python/ttnn_basic_operations.py
   2025-06-23 09:47:12.093 | INFO     | __main__:main:19 -
   --- TT-NN Tensor Creation with Tiles (32x32) ---
   2025-06-23 09:47:12.117 | INFO     | __main__:main:47 - Tensor from fill value 1:
   tensor([[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]])
   2025-06-23 09:47:12.117 | INFO     | __main__:main:48 - Zeros:
   tensor([[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.bfloat16)
   2025-06-23 09:47:12.118 | INFO     | __main__:main:49 - Ones:
   tensor([[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]], dtype=torch.bfloat16)
   2025-06-23 09:47:12.119 | INFO     | __main__:main:50 - Random:
   tensor([[0.1367, 0.3320, 0.8125,  ..., 0.7969, 0.6250, 0.8906],
         [0.6914, 0.1377, 0.2480,  ..., 0.6406, 0.0109, 0.2080],
         [0.6992, 0.8750, 0.6133,  ..., 0.3086, 0.6562, 0.6016],
         ...,
         [0.1455, 0.8672, 0.0221,  ..., 0.3926, 0.1074, 0.9414],
         [0.5859, 0.1426, 0.8906,  ..., 0.5820, 0.0182, 0.7031],
         [0.8711, 0.1377, 0.7305,  ..., 0.4102, 0.2812, 0.6836]],
         dtype=torch.bfloat16)
   2025-06-23 09:47:12.120 | INFO     | __main__:main:51 - From expanded NumPy (TT-NN):
   tensor([[5., 5., 5.,  ..., 6., 6., 6.],
         [5., 5., 5.,  ..., 6., 6., 6.],
         [5., 5., 5.,  ..., 6., 6., 6.],
         ...,
         [7., 7., 7.,  ..., 8., 8., 8.],
         [7., 7., 7.,  ..., 8., 8., 8.],
         [7., 7., 7.,  ..., 8., 8., 8.]])
   2025-06-23 09:47:12.120 | INFO     | __main__:main:53 -
   --- TT-NN Tensor Operations on (32x32) Tiles ---
   2025-06-23 09:47:18.928 | INFO     | __main__:main:59 - Addition:
   tensor([[1.1406, 1.3359, 1.8125,  ..., 1.7969, 1.6250, 1.8906],
         [1.6953, 1.1406, 1.2500,  ..., 1.6406, 1.0078, 1.2109],
         [1.7031, 1.8750, 1.6172,  ..., 1.3125, 1.6562, 1.6016],
         ...,
         [1.1484, 1.8672, 1.0234,  ..., 1.3906, 1.1094, 1.9453],
         [1.5859, 1.1406, 1.8906,  ..., 1.5859, 1.0156, 1.7031],
         [1.8750, 1.1406, 1.7344,  ..., 1.4141, 1.2812, 1.6875]],
         dtype=torch.bfloat16)
   2025-06-23 09:47:18.929 | INFO     | __main__:main:62 - Element-wise Multiplication:
   tensor([[0.6836, 1.6641, 4.0625,  ..., 4.7812, 3.7500, 5.3438],
         [3.4531, 0.6875, 1.2422,  ..., 3.8438, 0.0654, 1.2500],
         [3.5000, 4.3750, 3.0625,  ..., 1.8516, 3.9375, 3.6094],
         ...,
         [1.0156, 6.0625, 0.1543,  ..., 3.1406, 0.8594, 7.5312],
         [4.0938, 1.0000, 6.2500,  ..., 4.6562, 0.1455, 5.6250],
         [6.0938, 0.9648, 5.1250,  ..., 3.2812, 2.2500, 5.4688]],
         dtype=torch.bfloat16)
   2025-06-23 09:47:18.930 | INFO     | __main__:main:65 - Matrix Multiplication:
   tensor([[17.5000, 13.4375, 16.7500,  ..., 15.2500, 13.0625, 17.2500],
         [17.5000, 13.4375, 16.7500,  ..., 15.2500, 13.0625, 17.2500],
         [17.5000, 13.4375, 16.7500,  ..., 15.2500, 13.0625, 17.2500],
         ...,
         [17.5000, 13.4375, 16.7500,  ..., 15.2500, 13.0625, 17.2500],
         [17.5000, 13.4375, 16.7500,  ..., 15.2500, 13.0625, 17.2500],
         [17.5000, 13.4375, 16.7500,  ..., 15.2500, 13.0625, 17.2500]],
         dtype=torch.bfloat16)
   2025-06-23 09:47:18.930 | INFO     | __main__:main:67 -
   --- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---
   2025-06-23 09:47:18.932 | INFO     | __main__:main:71 - Broadcast Add Result (TT-NN):
   tensor([[1.1406, 1.3359, 1.8125,  ..., 1.7969, 1.6250, 1.8906],
         [1.6953, 1.1406, 1.2500,  ..., 1.6406, 1.0078, 1.2109],
         [1.7031, 1.8750, 1.6172,  ..., 1.3125, 1.6562, 1.6016],
         ...,
         [1.1484, 1.8672, 1.0234,  ..., 1.3906, 1.1094, 1.9453],
         [1.5859, 1.1406, 1.8906,  ..., 1.5859, 1.0156, 1.7031],
         [1.8750, 1.1406, 1.7344,  ..., 1.4141, 1.2812, 1.6875]],
         dtype=torch.bfloat16)
