# MobileNetV3

**Platforms:** Wormhole (n150)
**Supported Input Resolution:** `(224, 224)` = (Height, Width)

## Introduction
MobileNetV3 is a family of lightweight convolutional neural networks designed for efficient mobile and embedded vision applications. It uses **Inverted Residual blocks** with **depthwise convolutions**, **Squeeze-and-Excitation modules**, and **H-swish activations** to deliver high accuracy with minimal compute.

This implementation adapts **MobileNetV3-Small** for Tenstorrent hardware using the TT-NN and TT-Metalium stack, optimized for throughput and low-latency inference on Wormhole devices.

This repository provides:
- A **reference PyTorch model** (via TorchVision) for correctness.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- **Tests**, **demo**, and **resources** (sample images).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
  - [Custom Images](#custom-images)
- [Performance](#performance)
- [Configuration Notes](#configuration-notes)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
  ```

## Repository Layout
```
models/
└── experimental/
    └── mobileNetV3/
        ├── resources/
        │   ├── dog.jpeg              # sample input image
        │   └── image_with_label.jpg  # sample output with label overlay
        │
        ├── tt/
        │   ├── custom_preprocessor.py
        │   ├── ttnn_invertedResidual.py
        │   ├── ttnn_mobileNetV3.py
        │   ├── ttnn_squeezeExcitation.py
        │   └── utils.py
        │
        ├── demo/
        │   └── mobilenetV3_demo.py
        │
        ├── runner/
        │   ├── performant_runner.py
        │   └── performant_runner_infra.py
        │
        ├── tests/
        │   ├── pcc/
        │   │   ├── common.py
        │   │   ├── test_mobilenetv3.py               # end-to-end pytest
        │   │   ├── test_mobilenetv3_multi_device.py  # multi-device pytest
        │   │   ├── test_ttnn_invertedResidual.py
        │   │   └── test_ttnn_squeezeExcitation.py
        │   ├── perf/
        │   │   ├── test_mobilenetv3_perf.py
        │   │
        │   └── test_stability.py
        │
        └── README.md
```

## Weights
MobileNetV3-Small weights are automatically downloaded from TorchVision when running the model. No manual download is required.

Note: The weights are the official ImageNet-pretrained weights (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`) from TorchVision.

## Quickstart
### Run Tests
```
pytest models/experimental/mobileNetV3/tests/pcc/test_mobilenetv3.py
```
This runs an end-to-end flow that:
  - Loads the MobileNetV3-Small Torch reference from TorchVision,
  - Runs the TT-NN graph,
  - Compares results (PCC validation).

### Run the Demo
```
python3 models/experimental/mobileNetV3/demo/mobilenetV3_demo.py
```

### Custom Images
Sample image(s) are placed under:
```
models/experimental/mobileNetV3/resources/
```
Then re-run either the demo:
```
python3 models/experimental/mobileNetV3/demo/mobilenetV3_demo.py
```

## Performance
### Single Device (BS=1):
- end-2-end perf with trace enable and 2CQ is `250` FPS

To run perf test:
```
pytest models/experimental/mobileNetV3/tests/perf/test_e2e_performant.py
```

### Multi-Device:
To run multi-device test:
```
pytest models/experimental/mobileNetV3/tests/perf/test_mobilenetv3_multi_device.py
```
This test validates MobileNetV3-Small on multiple devices using data parallelism with:
  - `ShardTensorToMesh` for input distribution across devices,
  - `ReplicateTensorToMesh` for weight replication,
  - `ConcatMeshToTensor` for output composition.


## Configuration Notes
- Resolution: (H, W) = (224, 224) is supported end-to-end.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.
- Batch Size: Demo/tests are written for BS=1. For larger BS you'll need to verify memory layouts and tile alignment.
- Memory Layouts: The TT-NN path uses ROW_MAJOR layout for resize ops and may pad channels to multiples of 32 to satisfy kernel/tile alignment.
- Weights: Auto-downloaded from TorchVision if not cached locally.
