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
- [Performance](#performance)
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
        │   ├── sample_input             # sample input images taken from nuscenes mini dataset
        │   ├── weight file(.pth)        # if not present will be downloaded
        │   └── sample_output            # will be generated once the demo is run
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
        │   ├── tt_cp_fpn.py
        │   ├── tt_grid_mask.py
        │   ├── tt_nms_free_coder.py
        │   ├── tt_petr_head.py
        │   ├── tt_petr_transformer.py
        │   ├── tt_petr.py
        │   ├── tt_positional_encoding.py
        │   ├── tt_vovnetcp.py
        │   ├── common.py
        │   ├── model_preprocessing.py
        │   └── utils.py
        ├── README.md
        ├── demo/
        │   ├── demo.py
        └── tests/
          ├── perf/
          │   ├── test_petr.py
          │   ├── test_petr_perf.py            # Device perf test
          │   └── test_petr_perf_e2e.py        # E2E perf test with CQ=1 and use_trace=False
          └── pcc/
              └── test_cp_fpn.py
              └── test_petr_head.py
              └── test_petr_transformer.py
              └── test_petr.py                  # end-to-end pytest
              └── test_positional_encoding.py
              └── test_vovnetcp.py
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
models/experimental/petr/tests/pcc/test_petr.py
```
This runs an end-to-end flow that:
  - Loads the Torch reference,
  - Runs the TT-NN graph,
  - compares results.

### Run the Demo
```
python3 models/experimental/petr/demo/demo.py
```
### Custom Images
Sample nuScenes image(s) are placed under:
```
models/experimental/petr/resources/sample_input
```
Then re-run either the demo:
```
python3 models/experimental/petr/demo/demo.py
```
Note: In the current demo, the calibration needs to be corrected. Since we do not use the nuScenes dataset fully at the moment, approximate calibration values were used. As a result, the predicted images shown in the visualization may not be fully accurate.

## Performance

### Single Device (BS=1)
- Device perf is `2.36` FPS
- E2E perf without trace and CQ=1 is `0.8` FPS

### Run Device Perf Test
```
pytest models/experimental/petr/test/perf/test_petr_perf.py
```

### Run E2E Test (without trace, command queue = 1)
```
pytest models/experimental/petr/test/perf/test_petr_perf_e2e.py
```

## Configuration Notes
- Resolution: (H, W) = (320, 800) is supported end-to-end.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.
- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.
