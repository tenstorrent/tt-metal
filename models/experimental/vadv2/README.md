# VADV2

GPU and CPU evaluation metrics on nuscenes mini dataset are here : [https://drive.google.com/file/d/1p5ESawe79n4SPgt3ZCPO4fOQ4sxVufhU/view?usp=sharing]

Inference steps for both GPU and CPU : "To be updated soon"

## Current Progress

- **Transformer block** bring-up is **complete**.
- Achieved **PCC** of **0.99**.
- Issues have been created for the fallbacks.


##  Ongoing Work

- Bringup of the **head model** is still in progress.
- Additional integration and validation steps are pending.
- **Please note:** intermediate and unused code should be cleaned up


## Model Weights

To run the Transformer block, you must first download the pretrained weights.

### 🔗 Download Instructions

1. Go to the following Google Drive link:
[https://drive.google.com/file/d/1uufTSZMv9xOanBQiWy4vDrGEvxFDi0wR/view?usp=sharing]
2. Download the `weights.zip` file.
3. Unzip it inside the project root directory(`models/experimental/vadv2/`)


## How to Run the transformer Block

Use the following command to run the Transformer block test:

```
pytest tests/ttnn/integration_tests/vadv2/test_tt_transformer.py
```

## ToDo:

- **Full model** integration.
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
