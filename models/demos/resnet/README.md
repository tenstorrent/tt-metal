# How to Run
+ To run the demo use:
'pytest --disable-warnings models/demos/resnet/demo/demo.py::test_demo_sample[20-models/demos/resnet/demo/images/-device_params0]'
; where 20 is the batch size, and 'models/demos/resnet/demo/images/' is where the images are located. Our model supports batch size of 2 and 1 as well, however the demo focuses on batch size 20 which has the highest throughput among the three options. This demo includes preprocessing, postprocessing and inference time for batch size 20. The demo will run the images through the inference twice. First, to capture the compile time, and cache all the ops, Second, to capture the best inference time on TT hardware.

+ Our second demo is designed to run ImageNet dataset, run this with
'pytest --disable-warnings models/demos/resnet/demo/demo.py::test_demo_imagenet[20-160-device_params0]'
; again 20 refer to batch size here and 160 is number of iterations(batches), hence the model will process 160 batch of size 20, total of 3200 images.

Note that the first time the model is run, ImageNet images must be downloaded from huggingface and stored in  'models/demos/resnet/demo/images/'; therefore you need to login to huggingface using your token: 'huggingface-cli login'
To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

# Inputs
+ Inputs by defaults are provided from 'models/demos/resnet/demo/images/' which includes 20 images from ImageNet dataset. If you wish to modify the input images, modify the abovementioned command by replacing the path with the path to your images. Example:
'pytest --disable-warnings models/demos/resnet/demo/demo.py::test_demo_sample[20-path/to/your/images-device_params0]'.

+ You must put at least 20 images in your directory, and if more images located in your directory, 20 of them will randomly be picked. In this demo we assume images come from ImageNet dataset, if your images are from a different source you might have to modify the preprocessing part of the demo.

## Details

+ The entry point to metal resnet model is `ResNet` in `metalResNetBlock50.py`. The model picks up certain configs and weights from TorchVision pretrained model. We have used `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` version from TorchVision as our reference.
Our ImageProcessor on other hand is based on `microsoft/resnet-50` from huggingface.

+ For the second demo (ImageNet), the demo will load the images from ImageNet batch by batch. When executed, the first iteration (batch) will always be slower since the iteration includes the compilation as well. Afterwards, each iterations take only miliseconds. For exact performance measurements please check out the first demo.

## Performance

+ To obtain device performance, run `./tt_metal/tools/profiler/profile_this.py -c "pytest models/demos/resnet/tests/test_metal_resnet50_performant.py::test_run_resnet50_inference[False-LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_20-device_params0]"`
This will generate a CSV report under `<this repo dir>/generated/profiler/reports/ops/<report name>`. The report file name is logged in the run output.

+ For end-to-end performance, run `pytest models/demos/resnet/tests/test_perf_resnet.py::test_perf_trace_2cqs_bare_metal[20-0.0040-19-device_params0]`. This will generate a CSV with the timings and throughputs.
Expected end-to-end perf: For batch = 20, it is about `5500 fps` currently. This may vary machine to machine.
