# UNet

Here are the steps to run the functional model implementation of UNet:

 1. Run the following command to implement UNet functional model ```pytest tests/ttnn/integration_tests/unet/test_ttnn_unet.py```
 2. We have used fallback op of conv in some places.
 3. Currently, the UNet model gets low pcc, working on it to improve the pcc.