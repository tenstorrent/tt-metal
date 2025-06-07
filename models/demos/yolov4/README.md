# Yolov4 Demo

## Platforms:
    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

## How to run yolov4

### Model performant running with Trace+2CQ

#### For 320x320:
- end-2-end perf is 80 FPS
  ```bash
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
#### For 640x640:
- end-2-end perf is 30 FPS
  ```bash
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```


### Single Image Demo

- Use the following command to run the yolov4 with a giraffe image:

For 320x320:
  ```bash
  pytest models/demos/yolov4/demo.py::test_yolov4[device_params0-resolution0]
  ```
For 640x640:
  ```bash
  pytest models/demos/yolov4/demo.py::test_yolov4[device_params0-resolution1]
  ```
- The output file `ttnn_yolov4_prediction_demo.jpg` will be generated.

- Use the following command to run the yolov4 with different input image:
  ```bash
  pytest  --disable-warnings --input-path=<PATH_TO_INPUT_IMAGE> models/demos/yolov4/demo.py
  ```


### mAP Accuracy Test
- To be added soon

### Web Demo
- You may try the interactive web demo (35 FPS end-2-end) for 320x320 following the [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/README.md)
