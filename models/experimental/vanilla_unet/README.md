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

- The entry point to the vanilla unet is located at:`models/experimental/vanilla_unet/ttnn/ttnn_unet.py`
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
pytest models/experimental/vanilla_unet/test/test_e2e_performant.py::test_e2e_performant
```

## How to run demo
To run the demo, make sure to run the following command to create new folders,

```sh
mkdir models/experimental/vanilla_unet/demo/pred
mkdir models/experimental/vanilla_unet/demo/pred_image_set
mkdir models/experimental/vanilla_unet/demo/imageset
```

Download all imagedataset from in [drive](https://drive.google.com/drive/folders/1eaV-VR5_3AL5j21nTTaLdv2XyT-SfrOD?usp=sharing) and place it in models/experimental/vanilla_unet/demo/images and models/experimental/vanilla_unet/demo/imageset

#### To run single image demo,

```sh
pytest models/experimental/vanilla_unet/demo/demo.py::test_unet_demo_single_image
```
Output images will be saved in the models/experimental/vanilla_unet/demo/pred folder

#### To run demo for image dataset,

```sh
pytest models/experimental/vanilla_unet/demo/demo.py::test_unet_demo_imageset
```
Output images will be saved in the models/experimental/vanilla_unet/demo/pred_image_set folder

### Evaluation test:
- To be added soon
