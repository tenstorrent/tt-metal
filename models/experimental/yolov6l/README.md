# Yolov6l

## How to run

To run the inference, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

If running on Wormhole N300, the following environment variable needs to be set:

```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

Command to run the inference pipeline with random tensor of resolution 640x480:
```
pytest tests/ttnn/integration_tests/yolov6l/test_ttnn_yolov6l.py
```
Note: Model runs only with pretrained weights, which are auto-downloaded.
