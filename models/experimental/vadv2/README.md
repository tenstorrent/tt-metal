## VADV2

## Platforms:
    Wormhole (n150, n300)

## Introduction
VADv2 (Video-based Autonomous Driving version 2) is a state-of-the-art multi-modal 3D perception and prediction model designed for autonomous driving applications.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
- Use the following command to run the vadv2 test:
```
pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py
```

## Details
- The entry point to vadv2 model is in `models/experimental/vadv2/tt/tt_vad.py`.
- Model Type: VAD (Video-based Autonomous Driving) Tiny variant
- Input Resolution - (384,640) (Height,Width)
- Batch Size : 1
- Inference steps for both GPU and CPU : [https://docs.google.com/document/d/1mcqm_TXuZpPpvtnT19BNeKqP-ilQfGqSBGcEF_X9onk/edit?usp=sharing]
- GPU and CPU evaluation metrics on nuscenes mini dataset are here : [https://drive.google.com/file/d/1p5ESawe79n4SPgt3ZCPO4fOQ4sxVufhU/view?usp=sharing]

- ## Note:
    - The test focuses on verifying the raw model outputs and does not include validation of post-processing steps.
