# Unet Vanilla

## Platforms:
    Wormhole (n150, n300)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
   - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`


## How to run (480x640 resolution)

Use the following command to run the inference pipeline:

```sh
pytest --disable-warnings models/demos/vanilla_unet/tests/pcc/test_ttnn_unet.py::test_unet
```

### Model performant running with Trace+2CQ
Use the following command to run the e2e perf:

```sh
pytest --disable-warnings models/demos/vanilla_unet/tests/perf/test_e2e_performant.py::test_e2e_performant
```

## How to run demo
To run the demo, make sure to run the following command to create new folders,

```sh
mkdir models/demos/vanilla_unet/demo/imageset
```

Download all imagedataset from in [drive](https://drive.google.com/drive/folders/1eaV-VR5_3AL5j21nTTaLdv2XyT-SfrOD?usp=sharing) and place it in `models/demos/vanilla_unet/demo/imageset`

### Performant Demo with Trace+2CQ

### Single image

Use the following command to run the demo with single input:

```sh
pytest --disable-warnings models/demos/vanilla_unet/demo/demo.py::test_unet_demo_single_image
```
Output images will be saved in the `models/demos/vanilla_unet/demo/pred` folder

### Multiple images

Use the following command to run the demo with multiple inputs:
```sh
pytest models/demos/vanilla_unet/demo/demo.py::test_unet_demo_imageset
```
Output images will be saved in the `models/demos/vanilla_unet/demo/pred_image_set` folder

### Evaluation test:
To run the test of ttnn vs ground truth & torch vs ground truth, please follow the following command,

```sh
pytest --disable-warnings models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet
```

Note: If vanilla unet evaluation test fails with the error: `ValueError: Sample larger than population or is negative`
Try deleting the "imageset" folder in "models/experimental/segmentation_evaluation" directory and try running again.


### Details

- The entry point to the vanilla unet is located at:`models/demos/vanilla_unet/ttnn/ttnn_unet.py`
- Batch Size :1
- Supported Input Resolution - (480,640) (Height,Width)
- End-2-end perf with Trace+2CQs is 42 FPS
