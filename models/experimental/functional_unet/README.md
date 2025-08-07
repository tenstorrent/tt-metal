# Unet Shallow

## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

To run UNet Shallow for multiple iterations on single-chip at the best performance:

```sh
pytest --disable-warnings models/experimental/functional_unet/tests/test_unet_trace.py::test_unet_trace_2cq
```

To run UNet Shallow for multiple iterations on N300 and T3000 at the best performance:

```sh
pytest --disable-warnings models/experimental/functional_unet/tests/test_unet_trace.py::test_unet_trace_2cq_multi_device
````

Use `pytest models/experimental/functional_unet/tests/test_unet_model.py` to run the functional UNet Shallow model on a single-chip.

## Supported Hardware

- N150
- N300
- T3K

## Other Details

- Inputs by default are random data.
- Weights by default are random data. We apply optimizations to tensors (such as batch folding) and as such, we modify the weights to ensure we don't achieve inf values.
