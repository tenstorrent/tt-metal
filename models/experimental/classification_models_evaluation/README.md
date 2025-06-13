# Classification Evaluation

- Using `imagenet-1k` validation dataset.

## Evaluation Table

| Model        | Resolution | Batch Size | Samples | TTNN vs Ground Truth(Correct_predictions/Total_predictions) | Torch vs Ground Truth (Correct_predictions/Total_predictions) |
|--------------|------------|------------|---------|-------------------------------|-------------------------------|
| ViT          | (224, 224) | 8          | 512     | 81.25%               | 82.23%                 |
| ResNet50     | (224, 224) | 16         | 512     | 78.52%                 | 75.59%                |
| MobileNetV2  | (224, 224) | 8          | 512     | 68.36%                 | 65.62%                 |


## To run the test of ttnn vs ground truth, please follow the following commands:

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_vit_classification[wormhole_b0-tt_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_resnet50_classification[16-act_dtype0-weight_dtype0-device_params0-tt_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_mobilenetv2_classification[8-224-tt_model-device_params0]
 ```

## To run the test of torch vs ground truth, please follow the following commands:

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_vit_classification[wormhole_b0-torch_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_resnet50_classification[16-act_dtype0-weight_dtype0-device_params0-torch_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_mobilenetv2_classification[8-224-torch_model-device_params0]
 ```
