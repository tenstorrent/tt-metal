# Yolov4 Demo

## How to run yolov4

### Model performant running with Trace+2CQ

#### For 320x320:
- end-2-end perf is 120 FPS
  ```bash
  pytest models/demos/wormhole/yolov4/test_yolov4_performant.py::test_run_yolov4_trace_2cqs_inference[resolution0-True-1-act_dtype0-weight_dtype0-device_params0]
  ```
#### For 640x640:
- end-2-end perf is 50 FPS
  ```bash
  pytest models/demos/wormhole/yolov4/test_yolov4_performant.py::test_run_yolov4_trace_2cqs_inference[resolution1-True-1-act_dtype0-weight_dtype0-device_params0]
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
