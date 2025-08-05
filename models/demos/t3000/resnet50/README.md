---

# ResNet50 Demo

## Introduction
ResNet50 is a deep convolutional neural network architecture with 50 layers, designed to enable training of very deep networks by using residual learning to mitigate the vanishing gradient problem.

## Details

+ The entry point to the Metal ResNet model is `ResNet` in `ttnn_functional_resnet50.py`.
+ The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
+ Our ImageProcessor on the other hand is based on `microsoft/resnet-50` from huggingface.

## Demo

+ To run the demo use:
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/t3000/resnet50/demo/demo.py::test_demo_sample
```
- Where 16 is the batch size per device, and `models/demos/ttnn_resnet/demo/images/` is where the images are located.
- Our model supports batch size of 2 and 1 as well, however the demo focuses on batch size 16 per device which has the highest throughput among the three options.
This demo includes preprocessing, postprocessing and inference time for batch size 16 per device. The demo will run the images through the inference thrice. First, discover the optimal shard scheme. Second to capture the compile time, and cache all the ops. Third, to capture the best inference time on TT hardware.


+ Our second demo is designed to run ImageNet dataset, run this with
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/t3000/resnet50/demo/demo.py::test_demo_imagenet
```
- The 16 refers to batch size per device here and 100 is the number of iterations(batches), hence the model will process 100 batches of size 128, total of 12800 images per device.
- Note that the first time the model is run, ImageNet images must be downloaded from huggingface and stored in  `models/demos/ttnn_resnet/demo/images/`; therefore you need to login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## Performance

### T3000
#### End-to-End Performance
+ For end-to-end performance, run
```python
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/resnet50/tests/test_perf_e2e_resnet50.py::test_perf_trace_2cqs
```
+ This will generate a CSV with the timings and throughputs.
+ **Expected end-to-end perf**: For batch = 16 per device, or batch 128 in total, it is about `35,800 fps` currently. This may vary machine to machine.
