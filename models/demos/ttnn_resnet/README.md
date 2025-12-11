# ResNet50

## Platforms
- **T3000 (LoudBox/QuietBox)**: 8 devices in 1x8 mesh configuration
- **TG (Galaxy)**: 32 devices in 8x4 mesh configuration

## Introduction
ResNet50 is a deep convolutional neural network architecture with 50 layers, designed to enable training of very deep networks by using residual learning to mitigate the vanishing gradient problem.

Read more about it at:
- [docs.pytorch.org](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
- [huggingface.co/microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
    - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run

### Demo
Run the demo with sample images:
```bash
pytest --disable-warnings models.demos.ttnn_resnet.tests/test_demo.py::test_demo_sample
```
Where 16 is the batch size per device, and `models/demos/ttnn_resnet/demo/images/` is where the images are located.

Run with ImageNet dataset:
```bash
pytest --disable-warnings models.demos.ttnn_resnet.tests/test_demo.py::test_demo_imagenet
```
The 16 refers to batch size per device and 100 is the number of iterations (batches).
- **T3000**: Processes 100 batches of size 128 total (8 devices × 16 per device), total of 12,800 images
- **TG**: Processes 100 batches of size 512 total (32 devices × 16 per device), total of 51,200 images

## Testing

### End-to-End Performance
For end-to-end performance testing, run:
```bash
pytest models.demos.ttnn_resnet.tests/test_perf_e2e_resnet50.py::test_perf_trace_2cqs
```
This will generate a CSV with the timings and throughputs.

**Expected end-to-end performance:**
- **T3000**: For batch = 16 per device (128 total), approximately **35,800 fps**
- **TG**: For batch = 16 per device (512 total), approximately **96,800 fps**

Performance may vary machine to machine.

## Details

+ The entry point to the Metal ResNet model is `ResNet` in `ttnn_functional_resnet50.py`.
+ The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
+ Our ImageProcessor on the other hand is based on `microsoft/resnet-50` from huggingface.
+ Our model supports batch size of 2 and 1 as well, however the demo focuses on batch size 16 per device which has the highest throughput among the three options.

This demo includes preprocessing, postprocessing and inference time for batch size 16 per device. The demo will run the images through the inference thrice:
1. First, discover the optimal shard scheme
2. Second to capture the compile time, and cache all the ops
3. Third, to capture the best inference time on TT hardware

## Platform-Specific Directories

For single device and other platform-specific versions, please refer to:

### Single Device
[Grayskull](/models/demos/grayskull/resnet50/)

[Wormhole_B0](/models/demos/wormhole/resnet50/)
