## VADV2

## Platforms:
    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

## Introduction

VADv2 (Video-based Autonomous Driving version 2) is a state-of-the-art multi-modal 3D perception and prediction model designed for autonomous driving applications.

## Details

- The entry point to vadv2 model is in `models/experimental/vadv2/tt/tt_vad.py`.
- Model Type: VAD (Video-based Autonomous Driving) Tiny variant
- Input Resolution - (384,640) (Height,Width)
- Batch Size : 1
- Inference steps for both GPU and CPU : [https://docs.google.com/document/d/1mcqm_TXuZpPpvtnT19BNeKqP-ilQfGqSBGcEF_X9onk/edit?usp=sharing]
- GPU and CPU evaluation metrics on nuscenes mini dataset are here : [https://drive.google.com/file/d/1p5ESawe79n4SPgt3ZCPO4fOQ4sxVufhU/view?usp=sharing]

## How to run the Model

- ### Model Weights

   - To run the model, you must first download the pretrained weights.

- ### Download Instructions

    1. Go to the following Google Drive link:
    [https://drive.google.com/file/d/1uufTSZMv9xOanBQiWy4vDrGEvxFDi0wR/view?usp=sharing]
    2. Download the `weights.zip` file.
    3. Unzip it inside the project root directory(`models/experimental/vadv2/`)
    4. Use the following command to run the vadv2 test:
    ```
    pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py
    ```

- ### Note:
    - The test focuses on verifying the raw model outputs and does not include validation of post-processing steps.
