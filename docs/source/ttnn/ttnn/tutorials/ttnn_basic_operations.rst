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

   print("\n--- TT-NN Tensor Creation with Tiles (32x32) ---")
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


Full example and output
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
        [1., 1., 1.,  ..., 1., 1., 1.]])
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
   tensor([[0.7656, 0.8242, 0.4004,  ..., 0.2656, 0.1973, 0.2930],
        [0.7617, 0.0187, 0.8945,  ..., 0.7891, 0.1875, 0.8828],
        [0.8398, 0.6719, 0.5273,  ..., 0.1709, 0.3672, 0.8438],
        ...,
        [0.7695, 0.1118, 0.9961,  ..., 0.1758, 0.2207, 0.6250],
        [0.5391, 0.6602, 0.0033,  ..., 0.0845, 0.0630, 0.5273],
        [0.3340, 0.0104, 0.9062,  ..., 0.6836, 0.1367, 0.4746]],
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
   tensor([[1.7656, 1.8281, 1.3984,  ..., 1.2656, 1.1953, 1.2969],
        [1.7656, 1.0156, 1.8984,  ..., 1.7891, 1.1875, 1.8828],
        [1.8438, 1.6719, 1.5312,  ..., 1.1719, 1.3672, 1.8438],
        ...,
        [1.7734, 1.1094, 2.0000,  ..., 1.1797, 1.2188, 1.6250],
        [1.5391, 1.6641, 1.0000,  ..., 1.0859, 1.0625, 1.5312],
        [1.3359, 1.0078, 1.9062,  ..., 1.6875, 1.1406, 1.4766]],
       dtype=torch.bfloat16)
   Element-wise Multiplication:
   tensor([[3.8281, 4.1250, 2.0000,  ..., 1.5938, 1.1875, 1.7578],
        [3.8125, 0.0933, 4.4688,  ..., 4.7500, 1.1250, 5.3125],
        [4.1875, 3.3594, 2.6406,  ..., 1.0234, 2.2031, 5.0625],
        ...,
        [5.3750, 0.7812, 6.9688,  ..., 1.4062, 1.7656, 5.0000],
        [3.7812, 4.6250, 0.0227,  ..., 0.6758, 0.5039, 4.2188],
        [2.3438, 0.0728, 6.3438,  ..., 5.4688, 1.0938, 3.7969]],
       dtype=torch.bfloat16)
   Matrix Multiplication:
   tensor([[19.2500, 15.6250, 16.8750,  ..., 13.8125, 13.5000, 18.8750],
        [19.2500, 15.6250, 16.8750,  ..., 13.8125, 13.5000, 18.8750],
        [19.2500, 15.6250, 16.8750,  ..., 13.8125, 13.5000, 18.8750],
        ...,
        [19.2500, 15.6250, 16.8750,  ..., 13.8125, 13.5000, 18.8750],
        [19.2500, 15.6250, 16.8750,  ..., 13.8125, 13.5000, 18.8750],
        [19.2500, 15.6250, 16.8750,  ..., 13.8125, 13.5000, 18.8750]],
       dtype=torch.bfloat16)

   --- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---
   Broadcast Add Result (TT-NN):
   tensor([[1.7656, 1.8281, 1.3984,  ..., 1.2656, 1.1953, 1.2969],
        [1.7656, 1.0156, 1.8984,  ..., 1.7891, 1.1875, 1.8828],
        [1.8438, 1.6719, 1.5312,  ..., 1.1719, 1.3672, 1.8438],
        ...,
        [1.7734, 1.1094, 2.0000,  ..., 1.1797, 1.2188, 1.6250],
        [1.5391, 1.6641, 1.0000,  ..., 1.0859, 1.0625, 1.5312],
        [1.3359, 1.0078, 1.9062,  ..., 1.6875, 1.1406, 1.4766]],
       dtype=torch.bfloat16)
