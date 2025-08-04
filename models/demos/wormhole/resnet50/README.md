# ResNet50

## Platforms:
  Wormhole (n150, n300)

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

## Demo
+ To run the demo use:
```python
pytest --disable-warnings models/demos/wormhole/resnet50/demo/demo.py::test_demo_sample
```
Where 16 is the batch size per device, and `models/demos/ttnn_resnet/demo/images/` is where the images are located.

+ Our second demo is designed to run ImageNet dataset, run this with
```python
pytest --disable-warnings models/demos/wormhole/resnet50/demo/demo.py::test_demo_trace_with_imagenet
```
The 16 refers to batch size per device here and 100 is the number of iterations(batches), hence the model will process 100 batches of size 16 (32 for n300), total of 1600 (3200 for n300) images.

## Testing
### Wormhole Device Performance
+ To obtain device performance, run
```python
pytest models/demos/wormhole/resnet50/tests/test_perf_device_resnet50.py::test_perf_device
```
+ This will run the model for 4 times and generate CSV reports under `<this repo dir>/generated/profiler/reports/ops/<report name>`. The report file name is logged in the run output.
+ It will also show a sumary of the device throughput in the run output.

### Wormhole End-to-End Performance
+ For end-to-end performance, run
```python
pytest models/demos/wormhole/resnet50/tests/test_perf_e2e_resnet50.py::test_perf_trace_2cqs
```
+ This will generate a CSV with the timings and throughputs.
+ **Expected end-to-end perf**: For batch = 16, it is about `4,700 fps` currently. This may vary machine to machine.

## Details
+ The entry point to the Metal ResNet model is `ResNet` in `ttnn_functional_resnet50.py`.
+ The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
+ Our ImageProcessor on the other hand is based on `microsoft/resnet-50` from huggingface.

- Our model supports batch size of 2 and 1 as well, however the demo focuses on batch size 16 per device which has the highest throughput among the three options.
This demo includes preprocessing, postprocessing and inference time for batch size 16 per device. The demo will run the images through the inference thrice. First, discover the optimal shard scheme. Second to capture the compile time, and cache all the ops. Third, to capture the best inference time on TT hardware.
