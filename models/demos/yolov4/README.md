# Yolov4 Demo

## How to run yolov4

- Use the following command to run the yolov4 performant impelementation (95 FPS):
  ```
  pytest models/demos/wormhole/yolov4/test_yolov4_performant.py::test_run_yolov4_trace_2cqs_inference[True-1-act_dtype0-weight_dtype0-device_params0]
  ```

- You may try the interactive web demo following this [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/web_demo/README.md) (25-30 FPS). NOTE: The post-processing is currently running on host. It will be moved to device soon which should significantly improve the end to end FPS.


- Use the following command to run a single-image demo for visualization. NOTE: the following demos are intented for visualization. It is not the performant implementation yet. And, the post processing is currently done on host which we will be moving to device soon.

- Use the following command to run the yolov4 with a giraffe image:
  ```
  pytest models/demos/yolov4/demo/demo.py
  ```

- Use the following command to run the yolov4 with different input image:
  ```
  pytest  --disable-warnings --input-path=<PATH_TO_INPUT_IMAGE> models/demos/yolov4/demo/demo.py
  ```

Once you run the command, The output file named `ttnn_prediction_demo.jpg` will be generated.
