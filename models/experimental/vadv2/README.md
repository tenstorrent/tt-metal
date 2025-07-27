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
- GPU and CPU evaluation metrics on nuscenes mini dataset are here : [https://drive.google.com/file/d/1p5ESawe79n4SPgt3ZCPO4fOQ4sxVufhU/view?usp=sharing]
- Inference steps for both GPU and CPU : "To be updated soon"

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
    pytest models/experimental/vadv2/tests/pcc/test_vad.py
    ```

- ### Note:
    - The test focuses on verifying the raw model outputs and does not include validation of post-processing steps.
    - For reference, here are the shapes of the raw outputs obtained on CPU before any post-processing:
        - bev_embed: torch.Size([10000, 1, 256]),
        - all_cls_scores: torch.Size([3, 1, 300, 10]),
        - all_bbox_preds: torch.Size([3, 1, 300, 10]),
        - all_traj_preds: torch.Size([3, 1, 300, 6, 12]),
        - all_traj_cls_scores: torch.Size([3, 1, 300, 6]),
        - map_all_cls_scores: torch.Size([3, 1, 100, 3]),
        - map_all_bbox_preds: torch.Size([3, 1, 100, 4]),
        - map_all_pts_preds: torch.Size([3, 1, 100, 20, 2]),
        - ego_fut_preds: torch.Size([1, 3, 6, 2])


- **Please note:** intermediate and unused code should be cleaned up
