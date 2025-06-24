.. MLP inference with TT-NN:

MLP inference with TT-NN
########################

In this example we will combine insight from the previous examples, and use TT-NN with PyTorch
to perform a simple MLP inference task. This will demonstrate how to use TT-NN for tensor
operations and model inference.

Lets create the example file,
``ttnn_mlp_inference_mnist.py``

Import the necessary libraries
------------------------------

Amongst these, torchvision is used to load the MNIST dataset, and ttnn is used for tensor operations on the Tenstorrent device.

.. code-block:: python

   import torch
   import torchvision
   import torchvision.transforms as transforms
   import numpy as np
   import ttnn
   from loguru import logger

Open Tenstorrent device
-----------------------

Create necessary device on which we will run our program.

.. code-block:: python

   # Open Tenstorrent device
   device = ttnn.open_device(device_id=0)


Load MNIST Test Data
--------------------

Load and convert the MNIST 28x28 grayscale images to tensors and normalize them.  Subsequently, lets create a DataLoader to iterate through the dataset.
This will allow us to perform inference on each image in the dataset.

.. code-block:: python

   # Load MNIST data
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
   testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

Load Pretrained MLP Weights
---------------------------

Load the pretrained MLP weights from a file.

.. code-block:: python

   # Pretrained weights
   weights = torch.load("mlp_mnist_weights.pt")
   W1 = weights["W1"]
   b1 = weights["b1"]
   W2 = weights["W2"]
   b2 = weights["b2"]
   W3 = weights["W3"]
   b3 = weights["b3"]

Basic accuracy tracking, inference, loop, and image flattening
--------------------------------------------------------------

Loop through the first 5 images in the data set, and convert the image from 1x28x28 to 1x784 by flattening it.
This is done to match the input shape of the MLP model.

.. code-block:: python

   correct = 0
   total = 0

   for i, (image, label) in enumerate(testloader):
      if i >= 5:
         break

         image = image.view(1, -1).to(torch.float32)

Convert to TT-NN Tensor
-----------------------

Convert the PyTorch tensor to TT-NN format with bfloat16 data type and TILE_LAYOUT.
This is necessary for efficient computation on the Tenstorrent device.

.. code-block:: python

   # Input tensor
   image_tt = ttnn.from_torch(image, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   image_tt = ttnn.to_layout(image_tt, ttnn.TILE_LAYOUT)

Layer 1 (Linear + ReLU)
-----------------------

Transposed weights are used to match TT-NN's expected shape. Bias reshaped to 1x128 for broadcasting, and compute output 1.

.. code-block:: python

   # Layer 1
   W1_tt = ttnn.from_torch(W1.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   W1_tt = ttnn.to_layout(W1_tt, ttnn.TILE_LAYOUT)
   b1_tt = ttnn.from_torch(b1.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   b1_tt = ttnn.to_layout(b1_tt, ttnn.TILE_LAYOUT)
   out1 = ttnn.linear(image_tt, W1_tt, bias=b1_tt)
   out1 = ttnn.relu(out1)

Layer 2 (Linear + ReLU)
-----------------------

Same pattern as Layer 1, but with different weights and biases.

.. code-block:: python

   # Layer 2
   W2_tt = ttnn.from_torch(W2.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   W2_tt = ttnn.to_layout(W2_tt, ttnn.TILE_LAYOUT)
   b2_tt = ttnn.from_torch(b2.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   b2_tt = ttnn.to_layout(b2_tt, ttnn.TILE_LAYOUT)
   out2 = ttnn.linear(out1, W2_tt, bias=b2_tt)
   out2 = ttnn.relu(out2)

Layer 3 (Output Layer)
----------------------

Final layer with 10 output (for digits 0-9). No ReLU activation here, as this is the output layer.

.. code-block:: python

   # Layer 3
   W3_tt = ttnn.from_torch(W3.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   W3_tt = ttnn.to_layout(W3_tt, ttnn.TILE_LAYOUT)
   b3_tt = ttnn.from_torch(b3.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
   b3_tt = ttnn.to_layout(b3_tt, ttnn.TILE_LAYOUT)
   out3 = ttnn.linear(out2, W3_tt, bias=b3_tt)

Convert Back to PyTorch and sum results
---------------------------------------

Final layer with 10 output (for digits 0-9). No ReLU activation here, as this is the output layer.

.. code-block:: python

   # Convert result back to torch
   prediction = ttnn.to_torch(out3)
   predicted_label = torch.argmax(prediction, dim=1).item()

   correct += predicted_label == label.item()
   total += 1

   logger.info(f"Sample {i+1}: Predicted={predicted_label}, Actual={label.item()}")

Full example and output
-----------------------

Lets put everything together in a complete example that can be run directly. This example will open a Tenstorrent device, create two tensors, perform the addition, and log the output tensor.
You can run the provided ``train_and_export_mlp.py`` script to generate the weights to a file named ``mlp_mnist_weights.pt``.

.. literalinclude:: ttnn_tutorials_basic_python/ttnn_mlp_inference_mnist.py
   :caption: Example Source Code

.. literalinclude:: ttnn_tutorials_basic_python/train_and_export_mlp.py
   :caption: Script to generate weights for example

Running this script will output the input tensors and the result of their addition, which should be a tensor filled with 3s. As shown below

.. code-block:: console

   2025-06-23 09:51:47.723 | INFO     | __main__:main:17 -
   --- MLP Inference Using TT-NN on MNIST ---
   2025-06-23 09:52:10.480 | INFO     | __main__:main:85 - Sample 1: Predicted=7, Actual=7
   2025-06-23 09:52:10.491 | INFO     | __main__:main:85 - Sample 2: Predicted=2, Actual=2
   2025-06-23 09:52:10.499 | INFO     | __main__:main:85 - Sample 3: Predicted=1, Actual=1
   2025-06-23 09:52:10.506 | INFO     | __main__:main:85 - Sample 4: Predicted=0, Actual=0
   2025-06-23 09:52:10.514 | INFO     | __main__:main:85 - Sample 5: Predicted=4, Actual=4
   2025-06-23 09:52:10.514 | INFO     | __main__:main:87 -
   TT-NN MLP Inference Accuracy: 5/5 = 100.00%
