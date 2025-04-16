# VGG Unet

## How to run

To run the inference, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

Command to run the inference pipeline with random weights and random tensor:

```sh
pytest tests/ttnn/integration_tests/vgg_unet/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_false]
```

To use the model with the trained weights, follow these steps:

- Download the weights from this [link](https://drive.google.com/file/d/1XZi_W5Pj4jLSI31WUAlYf0SWQMu0wL6X/view).

- Place the downloaded file in the models/experimental/functional_vgg_unet directory.

- Set the use_pretrained_weight option to True.

If running on Wormhole N300, the following environment variable needs to be set:

```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

Execute the following command:

```sh
pytest tests/ttnn/integration_tests/vgg_unet/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_true]
```

Use the following command to run the e2e perf:
`pytest models/experimental/functional_vgg_unet/tests/test_perf_vgg_unet.py::test_vgg_unet`

Use the following command to run the e2e perf with trace 2cq:
`pytest models/experimental/functional_vgg_unet/tests/test_e2e_performant.py`
