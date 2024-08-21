---

# ResNet50 Demo

## Introduction
ResNet50 is a deep convolutional neural network architecture with 50 layers, designed to enable training of very deep networks by using residual learning to mitigate the vanishing gradient problem.

The demo is a work in progress. 

## Details

+ The entry point to the Metal ResNet model is `ResNet` in `ttnn_functional_resnet50_new_conv_api.py`.
+ The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
+ Our ImageProcessor on the other hand is based on `microsoft/resnet-50` from huggingface.

## Performance

### Single Device

#### Device Performance
+ To obtain device performance, run `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./tt_metal/tools/profiler/profile_this.py -c "pytest models/demos/ttnn_resnet/tests/test_ttnn_resnet50_performant.py::test_run_resnet50_inference[16z-act_dtype0-weight_dtype0-math_fidelity0-device_params0]"`
+ This will generate a CSV report under `<this repo dir>/generated/profiler/reports/ops/<report name>`. The report file name is logged in the run output.

#### End-to-End Performance
+ For end-to-end performance, run `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/ttnn_resnet/tests/test_perf_ttnn_resnet.py::test_perf_trace_2cqs_bare_metal[16-0.004-25-device_params0]`. 
+ This will generate a CSV with the timings and throughputs.
+ **Expected end-to-end perf**: For batch = 16, it is about `4300 fps` currently. This may vary machine to machine.

### T3000
#### End-to-End Performance
+ For end-to-end performance, run `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/ttnn_resnet/tests/multi_device/test_perf_ttnn_resnet.py::test_perf_trace_2cqs_t3000[wormhole_b0-True-16-True-0.0043-60-device_params0]`. 
+ This will generate a CSV with the timings and throughputs.
+ **Expected end-to-end perf**: For batch = 16 per device, or batch 128 in total, it is about `31,700 fps` currently. This may vary machine to machine.