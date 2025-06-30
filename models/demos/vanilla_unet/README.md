# Unet Vanilla

## Platforms:
    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```
To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Details

- The entry point to the vanilla unet is located at:`models/demos/vanilla_unet/ttnn/ttnn_unet.py`
- Batch Size :1
- Supported Input Resolution - (480,640) (Height,Width)

## How to run (480x640 resolution)

Command to run the inference pipeline with random tensor:

```sh
pytest tests/ttnn/integration_tests/vanilla_unet/test_ttnn_unet.py
```

### Model performant running with Trace+2CQ
Use the following command to run the e2e perf:
- end-2-end perf is 42 FPS
```sh
pytest models/demos/vanilla_unet/test/test_e2e_performant.py::test_e2e_performant
```

## How to run demo
To run the demo, make sure to run the following command to create new folders,

```sh
mkdir models/demos/vanilla_unet/demo/pred
mkdir models/demos/vanilla_unet/demo/pred_image_set
mkdir models/demos/vanilla_unet/demo/imageset
```

Download all imagedataset from in [drive](https://drive.google.com/drive/folders/1eaV-VR5_3AL5j21nTTaLdv2XyT-SfrOD?usp=sharing) and place it in models/demos/vanilla_unet/demo/images and models/demos/vanilla_unet/demo/imageset

### Performant Demo with Trace+2CQ

### Single image

```sh
pytest models/demos/vanilla_unet/demo/demo.py::test_unet_demo_single_image
```
Output images will be saved in the models/demos/vanilla_unet/demo/pred folder

### Multiple images
```sh
pytest models/demos/vanilla_unet/demo/demo.py::test_unet_demo_imageset
```
Output images will be saved in the models/demos/vanilla_unet/demo/pred_image_set folder

### Evaluation test:
To run the test of ttnn vs ground truth, please follow the following command,

```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet[res0-device_params0-tt_model]
```

Note: If vanilla unet evaluation test fails with the error: `ValueError: Sample larger than population or is negative`
Try deleting the "imageset" folder in "models/experimental/segmentation_evaluation" directory and try running again.
