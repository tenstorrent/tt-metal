# Segmentation Evaluation

- Using LGG Segmentation Dataset
- Loading the dataset using kagglehub package

## The below observations are F1 scores for ttnn_model vs dataset(ground truth data):

- Vanilla Unet(320x320 resolution) for 18 images: 0.6739
- VGG Unet (256x256 resolution) for 15 images (with pretrained weights): 0.9002
- VGG Unet (256x256 resolution) for 15 images (without pretrained weights): 0.0535

To run the test of ttnn vs ground truth, please follow the following commands:

### Vanilla Unet (320x320):
```
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-tt_model]
```

### VGG Unet (256x256):

#### Without pretrained weights:
```
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_false-tt_model]
```

#### With pretrained weights:
The VGG Unet evaluation runs with random weights by default. To use real weights, download them as described below and ensure that `use_pretrained_weight` is set to True.

To use the model with the trained weights, follow these steps:
- Download the weights from this [link](https://drive.google.com/file/d/1XZi_W5Pj4jLSI31WUAlYf0SWQMu0wL6X/view).
- Place the downloaded file in the models/demos/vgg_unet directory.
- Set the use_pretrained_weight option to True.

Execute the following command:
```
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_true-tt_model]
```

## The below observations are F1 scores for torch_model vs dataset(ground truth data):

- Vanilla Unet(320x320 resolution) for 18 images: 0.6712
- VGG Unet (256x256 resolution) for 15 images (with pretrained weights): 0.9078
- VGG Unet (256x256 resolution) for 15 images (without pretrained weights): 0.1629

To run the test of torch vs ground truth, please follow the following commands:

### Vanilla Unet (320x320):
```
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-torch_model]
```

### VGG Unet (256x256):

#### Without pretrained weights:
```
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_false-torch_model]
```

#### With pretrained weights:
The VGG Unet evaluation runs with random weights by default. To use real weights, download them as described below and ensure that `use_pretrained_weight` is set to True.

To use the model with the trained weights, follow these steps:
- Download the weights from this [link](https://drive.google.com/file/d/1XZi_W5Pj4jLSI31WUAlYf0SWQMu0wL6X/view).
- Place the downloaded file in the models/demos/vgg_unet directory.
- Set the use_pretrained_weight option to True.

Execute the following command:
```
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet[device_params0-res0-pretrained_weight_true-torch_model]
```
