---

# ResNet50 Demo

## Introduction
ResNet50 is a deep convolutional neural network architecture with 50 layers, designed to enable training of very deep networks by using residual learning to mitigate the vanishing gradient problem.

## Details

+ The entry point to the Metal ResNet model is `ResNet` in `ttnn_functional_resnet50_new_conv_api.py`.
+ The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
+ Our ImageProcessor on the other hand is based on `microsoft/resnet-50` from huggingface.

## Demo

+ To run the demo use:
`pytest --disable-warnings models/demos/ttnn_resnet/demo/demo.py::test_demo_sample[16-models/demos/resnet/demo/images/-device_params0]`
; where 16 is the batch size, and `models/demos/resnet/demo/images/` is where the images are located. Our model supports batch size of 2 and 1 as well, however the demo focuses on batch size 16 which has the highest throughput among the three options. This demo includes preprocessing, postprocessing and inference time for batch size 16. The demo will run the images through the inference thrice. First, discover the optimal shard scheme. Second to capture the compile time, and cache all the ops. Third, to capture the best inference time on TT hardware.

+ Our second demo is designed to run ImageNet dataset, run this with
`pytest --disable-warnings models/demos/ttnn_resnet/demo/demo.py::test_demo_imagenet[16-100-device_params0]`; again 16 refer to batch size here and 100 is number of iterations(batches), hence the model will process 100 batch of size 16, total of 1600 images.

Note that the first time the model is run, ImageNet images must be downloaded from huggingface and stored in  `models/demos/resnet/demo/images/`; therefore you need to login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens


## Performance

### Single Device

#### Device Performance
+ To obtain device performance, run `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./tt_metal/tools/profiler/profile_this.py -c "pytest models/demos/ttnn_resnet/tests/test_ttnn_resnet50_performant.py::test_run_resnet50_inference[16z-act_dtype0-weight_dtype0-math_fidelity0-device_params0]"`
+ This will generate a CSV report under `<this repo dir>/generated/profiler/reports/ops/<report name>`. The report file name is logged in the run output.

#### End-to-End Performance
+ For end-to-end performance, run `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/ttnn_resnet/tests/test_perf_ttnn_resnet.py::test_perf_trace_2cqs_bare_metal[16-0.004-25-device_params0]`.
+ This will generate a CSV with the timings and throughputs.
+ **Expected end-to-end perf**: For batch = 16, it is about `4100 fps` currently. This may vary machine to machine.

### T3000
#### End-to-End Performance
+ For end-to-end performance, run `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/ttnn_resnet/tests/multi_device/test_perf_ttnn_resnet.py::test_perf_trace_2cqs_t3000[wormhole_b0-True-16-True-0.0043-60-device_params0]`.
+ This will generate a CSV with the timings and throughputs.
+ **Expected end-to-end perf**: For batch = 16 per device, or batch 128 in total, it is about `31,250 fps` currently. This may vary machine to machine.
