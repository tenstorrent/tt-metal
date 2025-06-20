.. Create and Add Two Tensors Example:

Create and Add Two Tensors
##########################

We will review a simple example that demonstrates how to create two tensors and
add them together using TT-NN, a high-level Python API designed for developers
to run models like LLaMA, Mistral, Stable Diffusion, and more on Tenstorrent devices.

Lets create the example file,
``ttnn_add_tensors.py``

Import the necessary libraries
------------------------------

.. code-block:: python

   import ttnn

Open Tenstorrent device
-----------------------

Create necessary device on which we will run our program.

.. code-block:: python

   # Open Tenstorrent device
   device = ttnn.open_device(device_id=0)


Host Tensor Creation
--------------------

Create two TT-NN tensors, and initialize them with values 1 and 2 respectively.  The preferred shape of the tensors is (32, 32) which will match the hardware's tile size.

.. code-block:: python

   # Create two TT-NN tensors with TILE_LAYOUT
        tt_tensor1 =  tt_t2 = ttnn.full(
            shape=(32, 32),
            fill_value=1.0,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_tensor2 =  tt_t2 = ttnn.full(
            shape=(32, 32),
            fill_value=2.0,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

Perform the addition operation and convert back
-----------------------------------------------

Now we can perform the addition operation on the two TT-NN tensors and print out the result.

.. code-block:: python

   # Perform eltwise addition on the device
   tt_result = ttnn.add(tt_tensor1, tt_tensor2)

   # Print output tensor
   print("Output tensor:")
   print(tt_result)

Full example and output
-----------------------

Lets put everything together in a complete example that can be run directly. This example will open a Tenstorrent device, create two tensors, perform the addition, and print the output tensor.

.. literalinclude:: ttnn_tutorials_basic_python/ttnn_add_tensors.py
   :caption: Source Code

Running this script will output the input tensors and the result of their addition, which should be a tensor filled with 3s. As shown below

.. code-block:: console

   $ python3 $TT_METAL_HOME/ttnn/tutorials/basic_python/ttnn_add_tensors.py
   Input tensors:
   ttnn.Tensor([[ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
             [ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
             ...,
             [ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
             [ 1.00000,  1.00000,  ...,  1.00000,  1.00000]], shape=Shape([32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)
   ttnn.Tensor([[ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
             [ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
             ...,
             [ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
             [ 2.00000,  2.00000,  ...,  2.00000,  2.00000]], shape=Shape([32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)
   Output tensor:
   ttnn.Tensor([[ 3.00000,  3.00000,  ...,  3.00000,  3.00000],
             [ 3.00000,  3.00000,  ...,  3.00000,  3.00000],
             ...,
             [ 3.00000,  3.00000,  ...,  3.00000,  3.00000],
             [ 3.00000,  3.00000,  ...,  3.00000,  3.00000]], shape=Shape([32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)
