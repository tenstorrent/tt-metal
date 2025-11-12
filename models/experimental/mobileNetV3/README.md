# 🧩 MobileNetV3 

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Supported Device](https://img.shields.io/badge/device-Wormhole%20(n150)-blue)
![Precision](https://img.shields.io/badge/precision-BF16%2FFP16-green)
![Input Resolution](https://img.shields.io/badge/input-224x224-lightgrey)
![Status](https://img.shields.io/badge/status-Stable-brightgreen)

---

## 🔍 Introduction

**MobileNetV3** is a family of lightweight convolutional neural networks designed for efficient mobile and embedded vision applications.  
It combines **Depthwise Separable Convolutions**, **Squeeze-and-Excitation modules**, and **H-swish activations** to deliver high accuracy with minimal compute.  

The implementation of the **MobileNetV3** architecture follows closely the original paper. It is customizable and offers different configurations for building Classification, Object Detection and Semantic Segmentation backbones. It was designed to follow a similar structure to **MobileNetV2** and the two share common building blocks.

---

## 📘 Overview

This implementation adapts **MobileNetV3** for **Tenstorrent hardware**, optimized for throughput and low-latency inference on **Wormhole** device.
The two variants described on the paper: the Large and the Small, both are constructed using the same code with the only difference being their configuration which describes the number of blocks, their sizes, their activation functions etc.

The model is validated using internal test suites under `tests/`.

---

## :heavy_check_mark: Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- (Optional, for profiling) Build with profiler enabled:
  ```bash
  ./build_metal.sh --enable-profiler

---

## 🗂️ Repository Layout

| Directory | Purpose |
|------------|----------|
| `tt/` | Core Tenstorrent native modules of **MobileNetV3** |
| `demo/` | Demo scripts and visualization |
| `resources/` | Sample images for testing |
| `tests/` | Validation(PCC) and Performance test scripts |
| `runner/` | Standardized model execution framework (Work in-progress) |


The `mobilenetv3/` directory plugs into this structure, exposing inference, profiling, and test utilities consistent with other models in the repo.

---

## 🚀 Quickstart: Run MobileNetV3

### Run Tests
```
models/experimental/mobileNetV3/tests/pcc/test_mobilenetv3.py
```
This runs an end-to-end flow that:

  - Loads the Torch reference from Torchvision,

  - Runs the TT-NN graph,

  - Post-processes outputs,

  - Optionally compares results and saves artifacts.

### Run the Demo
```
python models/experimental/mobileNetV3/demo/mobileNetV3.py \
  --input  <path/to/image.png> \
  --output <path/to/output_dir>
```
### Custom Images
You can place your image(s) under:
```
models/experimental/mobileNetV3/resources/
```
Then re-run either the demo:
```
python models/experimental/mobileNetV3/demo/mobileNetV3.py
-i models/experimental/mobileNetV3/resources/input.png
-o models/experimental/mobileNetV3/resources
```

Expected output:
```
Demo completed. 
Predicted classification label overlaid and image/s saved in output directory.  
```

---

## 🧪 Validation

MobileNetV3 is verified against PyTorch and ONNX reference implementations for correctness. [TBD]

| Backend | Mean Abs Error | 
|----------|----------------|
| Torch vs ONNX | TBD |
| ONNX vs TTNN | TBD | 

---

## 🧮 Profiling & Debugging

Tenstorrent profiling tools provide detailed visibility into kernel and tensor operations.

Capture a short performance trace:
```bash
tt-trace capture --model mobilenetv3_large.ttnn --duration 5s
tt-analyze trace.json --view timeline
```

Refer to the [Profiling Guide](../../docs/profiling.md) for more usage patterns.

---

## ⚠️ Known Limitations (TO BE CONFIRMED)

- Depthwise separable convolutions partially fused in current TTIR path.  
- `hardswish` and `relu6` activations implemented as composed ops (minor numerical deviation).  
- Quantized (INT8) mode experimental.  
- Mixed-precision mode not yet supported.

---

## Performance
### Single Device (BS=1):

- end-2-end perf is [TBD] FPS

To run perf test:
```
pytest models/experimental/mobilenetv3/tests/perf/test_perf.py
```

To collect perf reports with the profiler, build with `--enable-profiler`

## Configuration Notes

- Resolution: (H, W) = (224, 224) is supported end-to-end.

- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.

- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.

- Memory Layouts: The TT-NN path uses ROW_MAJOR layout for resize ops and may pad channels to multiples of 32 to satisfy kernel/tile alignment.

- Weights: The loader maps Detectron/PDL keys → internal module keys. It will auto-download weights if missing via the included script.

---

## 🧩 Integration Example

Here’s how you can integrate MobileNetV3 with the shared Tenstorrent runtime:

**WORK IN PROGRESS** 

```python
from tt.runner import performant_runner

runner = MobileNetV3PerformantRunner()
runner.run()
runner.release()
```

---

## 🔗 References

- [MobileNetV3 Paper (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)  
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models/mobilenetv3.html)  
- [Tenstorrent Developer SDK Docs](https://tenstorrent.com/developer-docs)  
- [TTNN API Reference](../../docs/ttnn_reference.md)

---
