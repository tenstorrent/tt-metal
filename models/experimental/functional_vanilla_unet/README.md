# Unet Vanila

## How to run

To run the inference pipeline with random tensor:

```sh
pytest tests/ttnn/integration_tests/vanilla_unet/test_ttnn_unet.py
```

To run the demo, make sure to run the following command to create new folders,

```sh
mkdir models/experimental/functional_vanilla_unet/demo/pred
mkdir models/experimental/functional_vanilla_unet/demo/pred_image_set
mkdir models/experimental/functional_vanilla_unet/demo/images
mkdir models/experimental/functional_vanilla_unet/demo/imageset
```

Download all imagedataset from in [drive](https://drive.google.com/drive/folders/1eaV-VR5_3AL5j21nTTaLdv2XyT-SfrOD?usp=sharing) and place it in models/experimental/functional_vanilla_unet/demo/images and models/experimental/functional_vanilla_unet/demo/imageset

To run single image demo,

```sh
pytest models/experimental/functional_vanilla_unet/demo/demo.py::test_unet_demo_single_image
```

To run demo for image dataset,

```sh
pytest models/experimental/functional_vanilla_unet/demo/demo.py::test_unet_demo_imageset
```
