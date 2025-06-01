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

   import torch
   import ttnn

Open Tenstorrent device
-----------------------

Create necessary device on which we will run our program.

.. code-block:: python

   # Open Tenstorrent device
   device = ttnn.open_device(device_id=0)


Building a data movement kernel
-------------------------------

Create two PyTorch tensors, and initialize them with values 1 and 2 respectively.  The preferred shape of the tensors is (32, 32) which will match the hardware's tile size.

.. code-block:: python

   # Create two PyTorch tensors filled with 1s and 2s
   torch_tensor1 = torch.full((32, 32), 1.0, dtype=torch.float32)
   torch_tensor2 = torch.full((32, 32), 2.0, dtype=torch.float32)

Convert PyTorch tensors to TT-NN tensors
----------------------------------------

Convert the PyTorch tensors to TT-NN tensors with the desired data type and layout. In this case, we will use `bfloat16` as the data type and `TILE_LAYOUT` for the layout.

.. code-block:: python

   # Convert PyTorch tensors to TT-NN tensors with TILE_LAYOUT
   tt_tensor1 = ttnn.from_torch(
      torch_tensor1,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=device
   )

   tt_tensor2 = ttnn.from_torch(
      torch_tensor2,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=device
   )

Perform the addition operation and convert back
-----------------------------------------------

Now we can perform the addition operation on the two TT-NN tensors and convert the result back to a PyTorch tensor.

.. code-block:: python

   # Perform eltwise addition on the device
   tt_result = ttnn.add(tt_tensor1, tt_tensor2)

   # Convert the result back to a PyTorch tensor for inspection
   torch_result = ttnn.to_torch(tt_result)

Full example and output
-----------------------

Lets put everything together in a complete example that can be run directly. This example will open a Tenstorrent device, create two tensors, perform the addition, and print the output tensor.

.. code-block:: python

   import torch
   import ttnn

   def main():
      # Open Tenstorrent device
      device = ttnn.open_device(device_id=0)

      try:
         # Create two PyTorch tensors filled with 1s and 2s
         torch_tensor1 = torch.full((32, 32), 1.0, dtype=torch.float32)
         torch_tensor2 = torch.full((32, 32), 2.0, dtype=torch.float32)

         # Print input tensors
         print("Input tensors:")
         print(torch_tensor1)
         print(torch_tensor2)

         # Convert PyTorch tensors to TT-NN tensors with TILE_LAYOUT
         tt_tensor1 = ttnn.from_torch(
            torch_tensor1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
         )

         tt_tensor2 = ttnn.from_torch(
            torch_tensor2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
         )

         # Perform eltwise addition on the device
         tt_result = ttnn.add(tt_tensor1, tt_tensor2)

         # Convert the result back to a PyTorch tensor for inspection
         torch_result = ttnn.to_torch(tt_result)

         # Print output tensor
         print("Output tensor:")
         print(torch_result)

      finally:
         # Close Tenstorrent device
         ttnn.close_device(device)

   if __name__ == "__main__":
      main()

Running this script will output the input tensors and the result of their addition, which should be a tensor filled with 3s. As shown below

.. code-block:: console

   $ python3 ttnn_add_tensors.py
   Input tensors:
   tensor([[1., 1., 1.,  ..., 1., 1., 1.],
           [1., 1., 1.,  ..., 1., 1., 1.],
           [1., 1., 1.,  ..., 1., 1., 1.],
            ...,
           [1., 1., 1.,  ..., 1., 1., 1.],
           [1., 1., 1.,  ..., 1., 1., 1.],
           [1., 1., 1.,  ..., 1., 1., 1.]])
   tensor([[2., 2., 2.,  ..., 2., 2., 2.],
           [2., 2., 2.,  ..., 2., 2., 2.],
           [2., 2., 2.,  ..., 2., 2., 2.],
           ...,
           [2., 2., 2.,  ..., 2., 2., 2.],
           [2., 2., 2.,  ..., 2., 2., 2.],
           [2., 2., 2.,  ..., 2., 2., 2.]])
   Output tensor:
   tensor([[3., 3., 3.,  ..., 3., 3., 3.],
           [3., 3., 3.,  ..., 3., 3., 3.],
           [3., 3., 3.,  ..., 3., 3., 3.],
           ...,
           [3., 3., 3.,  ..., 3., 3., 3.],
           [3., 3., 3.,  ..., 3., 3., 3.],
         [3., 3., 3.,  ..., 3., 3., 3.]], dtype=torch.bfloat16)
                     Metal | INFO     | Closing device 0
                     Metal | INFO     | Disabling and clearing program cache on device 0

Source Code
###########
.. toctree::
   :maxdepth: 1
   :caption: Source Code

   ttnn_tutorials_python3/ttnn_add_tensors.py
