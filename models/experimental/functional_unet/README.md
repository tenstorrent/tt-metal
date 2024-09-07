# Unet Shallow

## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

Use `pytest --disable-warnings models/experimental/functional_unet/tests/test_unet_model.py` to run the full UNet Shallow model.

## Supported Hardware

- N150
- N300
  - Make sure to place dispatch on ethernet cores with `export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` for optimal performance

## Other Details

- Inputs by default are random data.
- Weights by default are random data. We apply optimizations to tensors (such as batch folding) and as such, we modify the weights to ensure we don't achieve inf values.
