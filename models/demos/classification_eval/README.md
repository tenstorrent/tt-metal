# Classification Evaluation

- Dataset used: `imagenet-1k` validation dataset. [source](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

## Evaluation Table

| Model        | Resolution | Batch Size | Samples | TTNN Accuracy | Torch Accuracy |
|--------------|------------|------------|---------|-------------------------------|-------------------------------|
| ViT          | (224, 224) | 8          | 512     | 81.25%               | 82.23%                 |
| ResNet50     | (224, 224) | 16         | 512     | 79.10%                 | 76.56%                |
| MobileNetV2  | (224, 224) | 10          | 512     | 69.40%                 | 71.80%                |
| VoVNet       | (224, 224) | 1          | 512     | 74.41%                 | 80.08%                 |
| EfficientNetB0| (224, 224) | 1          | 512     | 75.39%                | 76.76%                 |
| SwinV2       | (512, 512) | 1          | 512     | 75.59%                 | 81.05%                 |
| Swin_S       | (512, 512) | 1          | 512     | 81.05%                 | 82.23%                 |

***Note:*** The accuracy is for the selected random samples from the validation dataset.

Where,
- **TTNN Accuracy** refers to the ratio of correct predictions made by TTNN model to the total number of predictions, calculated by comparing TTNN outputs against the ground truth data(Labels given in validation dataset).
- **Torch Accuracy** refers to the ratio of correct predictions made by torch model to the total number of predictions, calculated by comparing Torch outputs against the ground truth data(Labels given in validation dataset).

## To run the test of ttnn vs ground truth, please follow the following commands:

 **SwinV2:** <br>
**_For 512x512,_**<br>

**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_v2_image_classification_eval[1-512-tt_model-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_v2_image_classification_eval_dp[wormhole_b0-1-512-tt_model-device_params0]
 ```

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_vit_image_classification_eval[wormhole_b0-tt_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_resnet50_image_classification_eval[16-act_dtype0-weight_dtype0-device_params0-tt_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>

**_Single-Device (BS-10):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval[tt_model-10-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval_dp[wormhole_b0-tt_model-10-device_params0]
 ```

**VoVNet:** <br>
**_For 224x224,_**<br>

**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_vovnet_image_classification_eval[1-224-tt_model-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_vovnet_image_classification_eval_dp[wormhole_b0-1-224-tt_model-device_params0]
 ```

**EfficientNetB0:** <br>
**_For 224x224,_**<br>
**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_efficientnetb0_image_classification_eval[1-224-tt_model-device_params0]
 ```
**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_efficientnetb0_image_classification_eval_dp[wormhole_b0-1-224-tt_model-device_params0]
 ```

**Swin_S:** <br>
**_For 512x512,_**<br>

**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_s_image_classification_eval[1-512-tt_model-device_params0]
 ```
**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_s_image_classification_eval_dp[wormhole_b0-1-512-tt_model-device_params0]
 ```
## To run the test of torch vs ground truth, please follow the following commands:

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_vit_image_classification_eval[wormhole_b0-torch_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_resnet50_image_classification_eval[16-act_dtype0-weight_dtype0-device_params0-torch_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>

**_Single-Device (BS-10):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval[torch_model-10-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval_dp[wormhole_b0-torch_model-10-device_params0]
 ```

**VoVNet:** <br>
**_For 224x224,_**<br>

**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_vovnet_image_classification_eval[1-224-torch_model-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_vovnet_image_classification_eval_dp[wormhole_b0-1-224-torch_model-device_params0]
 ```

**EfficientNetB0:** <br>
**_For 224x224,_**<br>

**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_efficientnetb0_image_classification_eval[1-224-torch_model-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_efficientnetb0_image_classification_eval_dp[wormhole_b0-1-224-torch_model-device_params0]
 ```

**SwinV2:** <br>
**_For 512x512,_**<br>

**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_v2_image_classification_eval[1-512-torch_model-device_params0]
 ```

**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_v2_image_classification_eval_dp[wormhole_b0-1-512-torch_model-device_params0]
 ```

**Swin_S:** <br>
**_For 512x512,_**<br>
**_Single-Device (BS-1):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_s_image_classification_eval[1-512-torch_model-device_params0]
 ```
**_Multi-Device (DP-2,N300):_**<br>
 ```sh
 pytest models/demos/classification_eval/classification_eval.py::test_swin_s_image_classification_eval_dp[wormhole_b0-1-512-torch_model-device_params0]
 ```
