# MobileNetV2 Model Using TTNN

This project implements the MobileNetV2 model using TTNN (Tenstorrent Neural Network Library) on Tenstorrent hardware without requiring LibTorch at runtime.

## Platforms
- WH N300

## Prerequisites
- TT-Metal environment set up
- MobileNetV2 exported weights generated in `models/experimental/mobilenet/mobilenetv2_cpp/script/mobilenet_v2-b0353104_weights`

## Getting Started

### 1. Generate Exported Weights
Use the provided script to download the pretrained MobileNetV2 weights and export the lightweight `manifest.json + .bin` parameter directory:

```bash
python3 models/experimental/mobilenet/mobilenetv2_cpp/script/gen_mobilenetv2_script.py
```

This generates `models/experimental/mobilenet/mobilenetv2_cpp/script/mobilenet_v2-b0353104_weights/`.

### 2. Build the Project
Build the targets:

```bash
./build_metal.sh
```

The generated binaries will be located at:
- `build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_e2e`: Performance-optimized implementation using Trace and 2 Command Queues.
- `build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_cpp`: Baseline implementation for functional execution.

### 3. Run Inference
Run the compiled binaries with the exported weights directory as the command-line argument:

```bash
./build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_e2e models/experimental/mobilenet/mobilenetv2_cpp/script/mobilenet_v2-b0353104_weights
./build/models/experimental/mobilenet/mobilenetv2_cpp/mobilenetv2_cpp models/experimental/mobilenet/mobilenetv2_cpp/script/mobilenet_v2-b0353104_weights
```

## Performance
The C++ implementation (`test/e2e_test.cpp`) is expected to match the performance behavior of the Python reference implementation (`models/experimental/mobilenetv2/tests/test_e2e_performant.py`).

### Sample Output
**Optimized Version (`mobilenetv2_e2e`)**:
```
mobilenetv2 batch_size=1, output_numel=1000
ttnn_mobilenetv2_224x224_batch_size_1. One inference iteration time (sec): 0.012440, FPS: 80.39
```

**Baseline Version (`mobilenetv2_cpp`)**:
```
mobilenetv2 batch_size=1, output_numel=1000
ttnn_mobilenetv2_224x224_batch_size_1. One inference iteration time (sec): 0.012440, FPS: 80.39
```
