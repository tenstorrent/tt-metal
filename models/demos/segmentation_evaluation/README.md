# Segmentation Evaluation
This folder contains evaluation scripts and results for the following segmentation models:
- Vanilla UNet
- VGG UNet
- Segformer

Each model is benchmarked using standard segmentation metrics:
- **IoU (Intersection over Union):** Measures the overlap between the predicted and actual regions.
- **Dice Score:** Measures the similarity between the predicted and actual segmentation.
- **Pixel Accuracy:** Percentage of correctly classified pixels.
- **Precision:** Percentage of predicted positives that are actually correct.
- **Recall:** Percentage of actual positives that were correctly predicted.
- **F1 Score:** Harmonic mean of precision and recall for balanced evaluation.

## To run the test of ttnn vs ground truth, please follow the following commands:

**VGG Unet (256x256):**

**_Single-Device (BS-1, N150 | P150):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-1-pretrained_weight_true-tt_model]
```

**_Multi-Device (DP-2,N300):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet_dp[wormhole_b0-device_params0-res0-1-pretrained_weight_true-tt_model]
```

**Segformer-b0 (512x512):**

**_Single-Device (BS-1):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_segformer_eval[res0-1-tt_model-device_params0]
```

**_Multi-Device (DP-2,N300):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_segformer_eval_dp[wormhole_b0-res0-1-tt_model-device_params0]
```

## To run the test of torch vs ground truth, please follow the following commands:

**Vanilla Unet (480x640):**

**_Single-Device (BS-1):_**<br>

```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-1-torch_model]
```

**_Multi-Device (DP-2,N300):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet_dp[wormhole_b0-res0-device_params0-1-torch_model]
```

**VGG Unet (256x256):**

**_Single-Device (BS-1):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-1-pretrained_weight_true-torch_model]
```

**_Multi-Device (DP-2,N300):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet_dp[wormhole_b0-device_params0-res0-1-pretrained_weight_true-torch_model]
```

**Segformer-b0 (512x512):**

**_Single-Device (BS-1):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_segformer_eval[res0-1-torch_model-device_params0]
```

**_Multi-Device (DP-2,N300):_**<br>
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_segformer_eval_dp[wormhole_b0-res0-1-torch_model-device_params0]
```

## Evaluation Table
| Model                 | Resolution | Dataset          | Samples      | IoU             | Dice Score      | Pixel Accuracy  | Precision       | Recall          | F1 Score        |
| --------------------- | ---------- | ---------------- | ------------ | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
|                       |            |                  | Torch / TTNN | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    |
| Vanilla Unet          | (480, 640) | LGG Segmentation | 523 / 523    | 50.02% / 47.55% | 61.47% / 58.43% | 99.41% / 99.39% | 81.92% / 77.63% | 53.36% / 50.92% | 61.47% / 58.43% |
| VGG Unet (Pretrained) | (256, 256) | LGG Segmentation | 500 / 500    | 81.78% / 80.23% | 89.38% / 88.25% | 99.53% / 99.48% | 84.43% / 82.24% | 96.37% / 96.90% | 89.38% / 88.25% |
| SegFormer-B0          | (512, 512) | ADE20K           | 2000 / 2000    | 89.93% / 89.18% | 5.95% / 6.02%   | 70.33% / 69.87% | 27.97% / 27.84% | 28.20% / 28.20% | 27.89% / 27.83% |

**Note:**
- The above metrics are calculated by comparing Torch/TTNN model vs dataset(ground truth data).

## Removed Models

### YOLOv9c
YOLOv9c segmentation evaluation was previously included in this folder but has been temporarily removed pending verification of usage rights for the YOLOv models. The evaluation code and tests for YOLOv9c have been disabled until we can verify proper licensing and usage permissions. Historical evaluation results for YOLOv9c (640x640 resolution on COCO 2017 dataset): IoU 27.57%/31.94%, Dice Score 37.71%/42.97% (Torch/TTNN, 18/199 samples).
