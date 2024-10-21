## INTRODUCTION
The MNIST model uses only fully connected linear layers to classify handwritten digits from the MNIST dataset. Despite the absence of convolutional layers, the model efficiently processes the 28x28 pixel images by flattening them into a 1D vector and passing them through multiple linear layers to predict the corresponding digit (0-9). This approach demonstrates how even simpler architectures can be applied for image classification tasks.

## How to Run

To run the demo for digit classification using the MNIST model, follow these instructions:

-  Use the following command to run the MNIST model.
  ```
  pytest models/demos/wormhole/mnist/demo/demo.py::test_demo_dataset
  ```

# Additional Information

The input tensor for reshape op is in the host.
