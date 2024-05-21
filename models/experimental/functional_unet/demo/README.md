The UNet inference is ran on a single input image, which gets pcc around 0.0 (i.e. pcc = -0.00910458441036667)
To run the demo:
 1. Take the branch harini/unet_inference
 2. Get the original weights file of Unet (i.e. unet.pt) from the UNet Brain MRI repository.
 3. Save the weights file in this directory tests/ttnn/integration_tests/unet/
 4. The input image for testing is added to the branch, save it to the directory models/experimental/functional_unet/demo/
 5. Run demo by the command: pytest models/experimental/functional_unet/demo/demo.py
