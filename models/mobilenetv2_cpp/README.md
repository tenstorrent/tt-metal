# Mobilenetv2 Model Using TTNN

## Platforms:
    WH N300

## Details
The C++ implementation `test/e2e_test.cpp` is expected to be as fast as that of the original `models/experimental/mobilenetv2/tests/test_e2e_performant.py` in performance and the generated binary file is `mobilenetv2_e2e`.

The program uses libtorch API to load the MobileNetV2 model. As libtorch only accepts TorchScript-format model file, this project converts the original MobileNetV2 model to TorchScript using `models/mobilenetv2_cpp/script/gen_mobilenetv2_script.py` , and all subsequent tests are carried out on the converted model file.

**Note:** You can specify the model file location by setting the `MOBILENET_FILE_PATH` environment variable, or the program will load MobileNetV2 model using file `${TT_METAL_HOME}/models/mobilenetv2_cpp/mobilenet_v2-b0353104-script.pt` by default.
