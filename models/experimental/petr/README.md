# PETR (Position Embedding TRansformer) (TT-NN)

**Platforms:** Wormhole (n150)
**Supported Input Resolution:** `(320, 800)` = (Height, Width)

## Introduction
PETR (Position Embedding TRansformer) is a transformer-based approach for multi-view 3D object detection that develops position embedding transformation to encode 3D coordinate information into image features. This implementation brings PETR to Tenstorrent hardware using the TT-NN and TT-Metalium stack.

This repository provides:
- A **reference PyTorch model** for correctness.
- A **TT-NN implementation** for Tenstorrent hardware (Wormhole).
- A **tests**, and **resources** (weights + sample assets).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Layout](#repository-layout)
- [Weights](#weights)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
  - [Custom Images](#custom-images)
- [Performance (Trace + 2CQ)](#performance-trace--2cq)
- [Configuration Notes](#configuration-notes)

## Prerequisites
- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- (Optional, for profiling) Build with profiler enabled:
  ```bash
  ./build_metal.sh --enable-profiler

## Repository Layout
```
models/
└── experimental/
    └── petr/
        ├── resources/
        │   ├──
        │   ├──
        │   ├──
        │   ├──
        │
        ├── reference/
        │   ├── cp_fpn.py
        │   ├── grid_mask.py
        │   ├── nms_free_coder.py
        │   ├── petr_head.py
        │   ├── petr_transformer.py
        │   ├── petr.py
        │   ├── positional_encoding.py
        │   ├── vovnetcp.py
        │   └── utils.py
        ├── tt/
        │   ├── ttnn_cp_fpn.py
        │   ├── ttnn_grid_mask.py
        │   ├── ttnn_nms_free_coder.py
        │   ├── ttnn_petr_head.py
        │   ├── ttnn_petr_transformer.py
        │   ├── ttnn_petr.py
        │   ├── ttnn_positional_encoding.py
        │   ├── ttnn_vovnetcp.py
        │   ├── common.py
        │   ├── model_preprocessing.py
        │   └── utils.py
        ├── README.md
        ├── demo/
        │   ├── demo.py
        └── tests/
          ├── perf/
          │   ├── test_ttnn_perf.py
          └── pcc/
              └── test_ttnn_cp_fpn.py
              └── test_ttnn_petr_head.py
              └── test_ttnn_petr_transformer.py
              └── test_ttnn_petr.py                 # end-to-end pytest
              └── test_ttnn_positional_encoding.py
              └── test_ttnn_vovnetcp.py
```
## Weights
The default model expects petr_vovnet_gridmask_p4_800x320-e2191752.pth in:
```
models/experimental/petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth
```
If missing, the code will download the same to the path
```

Note: The weights are for nuScenes dataset VoVNet backbone.
## Quickstart
### Run Tests
```
models/experimental/petr/tests/pcc/test_ttnn_petr.py
```
This runs an end-to-end flow that:
  - Loads the Torch reference,
  - Runs the TT-NN graph,
  - compares results.

### Run the Demo
```
TODO - To be filled
```
### Custom Images
You can place your image(s) under:
```
models/experimental/petr/resources/
```
Then re-run either the demo:
```
TODO - TO be filled
```
## Performance
## TODO
### Single Device (BS=1):
- end-2-end perf is `` FPS
To run perf test:
```
pytest models/experimental/petr/tests/perf/test_ttnn_perf.py
```
To collect perf reports with the profiler, build with `--enable-profiler`
## Configuration Notes
- Resolution: (H, W) = (320, 800) is supported end-to-end.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.
- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.
