# Yolov4 Demo

## How to run yolov4 (320x320 resolution)

### Model performant running with Trace+2CQ
- Use the following command to run the yolov4 performant implementation (end-2-end perf is 120 FPS):
  ```bash
  pytest models/demos/wormhole/yolov4/test_yolov4_performant_webdemo.py::test_run_yolov4_trace_2cqs_inference[True-1-act_dtype0-weight_dtype0-device_params0]
  ```
- The end-2-end 120 FPS corresponds to device-only runtime of 185 FPS


### Single Image Demo

- Use the following command to run the yolov4 with a giraffe image:
  ```bash
  pytest models/demos/yolov4/demo/demo.py
  ```
- The output file `ttnn_yolov4_320_prediction_demo.jpg` will be generated.

- Use the following command to run the yolov4 with different input image:
  ```bash
  pytest  --disable-warnings --input-path=<PATH_TO_INPUT_IMAGE> models/demos/yolov4/demo/demo.py
  ```


### mAP Accuracy Test
- To be added soon

### Web Demo
- You may try the interactive web demo (35 FPS end-2-end) following the [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/README.md)
