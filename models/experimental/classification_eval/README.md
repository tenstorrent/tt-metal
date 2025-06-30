# Classification Evaluation

- Using `imagenet-1k` validation dataset.

## Evaluation Table

| Model        | Resolution | Batch Size | Samples | TTNN Accuracy | Torch Accuracy |
|--------------|------------|------------|---------|-------------------------------|-------------------------------|
| ViT          | (224, 224) | 8          | 512     | 81.25%               | 82.23%                 |
| ResNet50     | (224, 224) | 16         | 512     | 78.52%                 | 75.59%                |
| MobileNetV2  | (224, 224) | 8          | 512     | 68.36%                 | 65.62%                 |

***Note:*** The accuracy is for the selected random samples from the validation dataset.

Where,
- **TTNN Accuracy** refers to the ratio of correct predictions made by TTNN model to the total number of predictions, calculated by comparing TTNN outputs against the ground truth data(Labels given in validation dataset).
- **Torch Accuracy** refers to the ratio of correct predictions made by torch model to the total number of predictions, calculated by comparing Torch outputs against the ground truth data(Labels given in validation dataset).

## To run the test of ttnn vs ground truth, please follow the following commands:

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_eval/classification_eval.py::test_vit_image_classification_eval[wormhole_b0-tt_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_eval/classification_eval.py::test_resnet50_image_classification_eval[16-act_dtype0-weight_dtype0-device_params0-tt_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval[8-224-tt_model-device_params0]
 ```

## To run the test of torch vs ground truth, please follow the following commands:

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_eval/classification_eval.py::test_vit_image_classification_eval[wormhole_b0-torch_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_eval/classification_eval.py::test_resnet50_image_classification_eval[16-act_dtype0-weight_dtype0-device_params0-torch_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval[8-224-torch_model-device_params0]
 ```
