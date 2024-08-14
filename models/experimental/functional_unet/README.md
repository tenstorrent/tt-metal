## Unet Shallow Demo
## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

Use `pytest --disable-warnings models/experimental/functional_unet/demo/demo.py::test_unet_demo` to run all variants of the demo.

Run demo with functional mode `pytest --disable-warnings models/experimental/functional_unet/tests/test_unet_shallow_functional.py::test_unet_pcc`

Run demo with model performance mode `pytest --disable-warnings models/experimental/functional_unet/tests/test_unet_shallow_performance.py::test_unet_model_performance`

Run demo with device performance mode `pytest --disable-warnings models/experimental/functional_unet/tests/test_unet_shallow_performance.py::test_unet_device_performance`

# Supported
- n150
- n300
  - Make sure to place dispatch on ethernet cores with `export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` for optimal performance

# Inputs
Inputs by default are random data.

# Weights
Weights by default are random data. We apply optimizations to tensors (such as batch folding) and as such, we modify the weights to ensure we don't achieve inf values.

# Details
The entry point to  functional_unet model is UNet in `models/experimental/functional_unet/tt/unet_shallow_ttnn.py`.
