# Unet Vanila

## How to run

To run the inference, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

Command to run the inference pipeline with random tensor:

```sh
pytest tests/ttnn/integration_tests/vanilla_unet/test_ttnn_unet.py
```
