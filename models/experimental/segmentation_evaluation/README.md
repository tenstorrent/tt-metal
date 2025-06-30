# Segmentation Evaluation

- Using LGG Segmentation Dataset
- Loading the dataset using kagglehub package

## Evaluation Table
| Model                     | Resolution | Dataset          | Samples | IoU             | Dice Score      | Pixel Accuracy  | Precision       | Recall          | F1 Score        |
| ------------------------- | ---------- | ---------------- | ------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
|                           |            |                  |         | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    | Torch / TTNN    |
| Vanilla Unet              | (320, 320) | LGG Segmentation | 18      | 50.51% / 50.82% | 67.12% / 67.39% | 98.51% / 98.52% | 83.39% / 83.64% | 56.16% / 56.42% | 67.12% / 67.39% |
| VGG Unet (Pretrained)     | (256, 256) | LGG Segmentation | 15      | 83.94% / 82.76% | 90.78% / 90.02% | 99.55% / 99.50% | 86.32% / 84.51% | 96.34% / 97.05% | 90.78% / 90.02% |
| VGG Unet (Not Pretrained) | (256, 256) | LGG Segmentation | 15      | 9.39% / 2.79%   | 16.29% / 5.35%  | 71.31% / 13.50% | 9.68% / 2.84%   | 78.47% / 72.49% | 16.29% / 5.35%  |

**Note:** The above metrics are calculated by comparing Torch/TTNN model vs dataset(ground truth data).

## To run the test of ttnn vs ground truth, please follow the following commands:

**Vanilla Unet (320x320):**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-tt_model]
```
Note: If vanilla unet evaluation test fails with the error: `ValueError: Sample larger than population or is negative`
Try deleting the "imageset" folder in "models/experimental/segmentation_evaluation" directory, and uncomment the line 20 in "models/experimental/segmentation_evaluation/dataset_download.py" and try running again.

**VGG Unet (256x256):**

**_Without pretrained weights:_**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_false-tt_model]
```

**_With pretrained weights:_**

The VGG Unet evaluation runs with random weights by default. To use real weights, download them as described below and ensure that `use_pretrained_weight` is set to True.

To use the model with the trained weights, follow these steps:
- Download the weights from this [link](https://drive.google.com/file/d/1XZi_W5Pj4jLSI31WUAlYf0SWQMu0wL6X/view).
- Place the downloaded file in the models/demos/vgg_unet directory.
- Set the use_pretrained_weight option to True.

Execute the following command:
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_true-tt_model]
```

## To run the test of torch vs ground truth, please follow the following commands:

**Vanilla Unet (320x320):**
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-torch_model]
```

**VGG Unet (256x256):**

**_Without pretrained weights:_**

```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_false-torch_model]
```

**_With pretrained weights:_**

The VGG Unet evaluation runs with random weights by default. To use real weights, download them as described below and ensure that `use_pretrained_weight` is set to True.

To use the model with the trained weights, follow these steps:
- Download the weights from this [link](https://drive.google.com/file/d/1XZi_W5Pj4jLSI31WUAlYf0SWQMu0wL6X/view).
- Place the downloaded file in the models/demos/vgg_unet directory.
- Set the use_pretrained_weight option to True.

Execute the following command:
```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_true-torch_model]
```
