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

## Details

- The entry point to the vanilla unet is located at:`models/demos/vanilla_unet/ttnn/ttnn_unet.py`
- Batch Size :1
- Supported Input Resolution - (480,640) (Height,Width)

## How to run (480x640 resolution)

- Command to run the inference pipeline with random tensor:

```sh
pytest --disable-warnings tests/ttnn/integration_tests/vanilla_unet/test_ttnn_unet.py
```

### Model performant running with Trace+2CQ

#### Single Device (BS=1):

Use the following command to run the e2e perf:

- end-2-end perf is 42 FPS

```sh
pytest --disable-warnings models/demos/vanilla_unet/test/test_e2e_performant.py::test_e2e_performant
```

#### Multi Device (DP=2, N300):

Use the following command to run the e2e perf:

- end-2-end perf is 63 FPS

```sh
pytest --disable-warnings models/demos/vanilla_unet/test/test_e2e_performant.py::test_e2e_performant_dp
```

## How to run demo

### Performant Demo with Trace+2CQ

#### Single image

```sh
pytest --disable-warnings models/demos/vanilla_unet/demo/demo.py::test_unet_demo_single_image
```

- Output images will be saved in the `models/demos/vanilla_unet/demo/pred` folder

#### Multiple images

To run the demo with multiple images, make sure to run the following command to create new folders,

```sh
mkdir models/demos/vanilla_unet/demo/imageset
```

- Download all imagedataset from in [drive](https://drive.google.com/drive/folders/1eaV-VR5_3AL5j21nTTaLdv2XyT-SfrOD?usp=sharing) and place it in `models/demos/vanilla_unet/demo/imageset`

##### Single Device (BS=1):

```sh
pytest --disable-warnings models/demos/vanilla_unet/demo/demo.py::test_unet_demo_imageset
```

Output images will be saved in the `models/demos/vanilla_unet/demo/pred_image_set` folder

##### Multi Device (DP=2, N300):

```sh
pytest --disable-warnings models/demos/vanilla_unet/demo/demo.py::test_unet_demo_imageset_dp
```
Output images will be saved in the `models/demos/vanilla_unet/demo/pred_image_set_dp` folder


### Performant Data evaluation with Trace+2CQ

#### Single Device (BS=1):

Use the following command to run the performant evaluation with Trace+2CQs:

```sh
pytest --disable-warnings models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet
```

#### Multi Device (DP=2, N300):

Use the following command to run the performant evaluation with Trace+2CQs:

```sh
pytest --disable-warnings models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet_dp
```

Note: If vanilla unet evaluation test fails with the error: `ValueError: Sample larger than population or is negative`
Try deleting the "imageset" folder in "models/experimental/segmentation_evaluation" directory and try running again.
