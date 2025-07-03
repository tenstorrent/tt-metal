# Segmentation Evaluation
This folder contains evaluation scripts and results for the following segmentation models:
- Vanilla UNet
- VGG UNet

Each model is benchmarked using standard segmentation metrics:
- **IoU (Intersection over Union):** Measures the overlap between the predicted and actual regions.
- **Dice Score:** Measures the similarity between the predicted and actual segmentation.
- **Pixel Accuracy:** Percentage of correctly classified pixels.
- **Precision:** Percentage of predicted positives that are actually correct.
- **Recall:** Percentage of actual positives that were correctly predicted.
- **F1 Score:** Harmonic mean of precision and recall for balanced evaluation.

## To run the test of ttnn vs ground truth, please follow the following commands:

**Vanilla Unet (320x320):**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-tt_model]
```
Note: If vanilla unet evaluation test fails with the error: `ValueError: Sample larger than population or is negative`
Try deleting the "imageset" folder in "models/experimental/segmentation_evaluation" directory, and uncomment the line 20 in "models/experimental/segmentation_evaluation/dataset_download.py" and try running again.

**VGG Unet (256x256):**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_true-tt_model]
```

## To run the test of torch vs ground truth, please follow the following commands:

**Vanilla Unet (320x320):**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-torch_model]
```

**VGG Unet (256x256):**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_true-torch_model]
```

## Evaluation Table
| Model                 | Resolution | Dataset          | Samples | IoU             | Dice Score      | Pixel Accuracy  | Precision       | Recall          | F1 Score        |
| --------------------- | ---------- | ---------------- | ------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
|                       |            |                  |         | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    |
| Vanilla Unet          | (480, 640) | LGG Segmentation | 523     | 50.02% / 47.55% | 61.47% / 58.43% | 99.41% / 99.39% | 81.92% / 77.63% | 53.36% / 50.92% | 61.47% / 58.43% |
| VGG Unet (Pretrained) | (256, 256) | LGG Segmentation | 500     | 81.78% / 80.23% | 89.38% / 88.25% | 99.53% / 99.48% | 84.43% / 82.24% | 96.37% / 96.90% | 89.38% / 88.25% |

**Note:** The above metrics are calculated by comparing Torch/TTNN model vs dataset(ground truth data).
