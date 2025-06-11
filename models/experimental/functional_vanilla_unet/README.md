# Unet Vanila

## How to run (480x640 resolution)

To run the inference, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

If running on Wormhole N300, the following environment variable needs to be set:

```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

Command to run the inference pipeline with random tensor:

```sh
pytest tests/ttnn/integration_tests/vanilla_unet/test_ttnn_unet.py
```

### E2E performant
Use the following command to run the e2e perf:
- end-2-end perf is 68.548 FPS
```sh
pytest models/experimental/functional_vanilla_unet/test/test_perf_vanilla_unet.py::test_vanilla_unet
```

## How to run demo
To run the demo, make sure to run the following command to create new folders,

```sh
mkdir models/experimental/functional_vanilla_unet/demo/pred
mkdir models/experimental/functional_vanilla_unet/demo/pred_image_set
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

## Supported Hardware

- N150
- N300
