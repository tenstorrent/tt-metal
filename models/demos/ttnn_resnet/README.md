---

# ResNet50 Demo

## Introduction
ResNet50 is a deep convolutional neural network architecture with 50 layers, designed to enable training of very deep networks by using residual learning to mitigate the vanishing gradient problem.

## Details

+ The entry point to the Metal ResNet model is `ResNet` in `ttnn_functional_resnet50_new_conv_api.py`.
+ The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
+ Our ImageProcessor on the other hand is based on `microsoft/resnet-50` from huggingface.

Please refer to the following directories for device specific versions and tests

### Single Device
[Grayskull](/models/demos/grayskull/resnet50/)

[Wormhole_B0](/models/demos/wormhole/resnet50/)

### Multi Device
[Wormhole_B0 T3000](/models/demos/t3000/resnet50/)

[Wormhole_B0 TG](/models/demos/tg/resnet50/)

[Wormhole_B0 TGG](/models/demos/tgg/resnet50/)
