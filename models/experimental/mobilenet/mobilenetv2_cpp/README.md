# MobileNetV2 Model Using TTNN

This project implements the MobileNetV2 model using TTNN (Tenstorrent Neural Network Library) and LibTorch for inference on Tenstorrent hardware.

## Platforms
- WH N300

## Prerequisites
- TT-Metal environment set up
- LibTorch installed

## Getting Started

### 1. Generate TorchScript Model
The C++ implementation uses LibTorch to load the model, which requires a TorchScript-formatted file. Use the provided script to download the pre-trained weights and generate the TorchScript model:

```bash
python3 models/experimental/mobilenet/mobilenetv2_cpp/script/gen_mobilenetv2_script.py
```
This will generate `mobilenet_v2-b0353104.pt` in `models/experimental/mobilenet/mobilenetv2_cpp/script/`.

### 2. Build the Project
Use CMake to build the target:

```bash
./build_metal.sh --enable-mobilenet-libtorch
```

The generated binaries will be located at:
- `build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_e2e`: Performance-optimized implementation using Trace and 2 Command Queues.
- `build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_cpp`: Baseline implementation for functional verification.

### 3. Run Inference
Run the compiled binary, providing the path to the generated TorchScript model as a command-line argument:

```bash
./build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_e2e models/experimental/mobilenet/mobilenetv2_cpp/script/mobilenet_v2-b0353104.pt
./build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_cpp models/experimental/mobilenet/mobilenetv2_cpp/script/mobilenet_v2-b0353104.pt
```

## Performance
The C++ implementation (`test/e2e_test.cpp`) is expected to match the performance of the Python reference implementation (`models/experimental/mobilenetv2/tests/test_e2e_performant.py`).

### Results
Below are the sample outputs for the two implementations:

**Optimized Version (`mobilenetv2_e2e`)**:
```
mobilenetv2 batch_size=1, PCC= 0.9520 - Passed
ttnn_mobilenetv2_224x224_batch_size_1. One inference iteration time (sec): 0.002176, FPS: 459.66,
```

**Baseline Version (`mobilenetv2_cpp`)**:
```
mobilenetv2 batch_size=1, PCC= 0.9528 - Passed
ttnn_mobilenetv2_224x224_batch_size_1. One inference iteration time (sec): 0.014258, FPS: 70.14
```
