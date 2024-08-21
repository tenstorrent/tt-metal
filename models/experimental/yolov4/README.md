# Yolov4 Demo

## How to run demo

- Use the following command to run the yolov4 with giraffe image from weka path:
  ```
  pytest models/experimental/yolov4/demo/demo.py
  ```

- Use the following command to run the yolov4 with different input image:
  ```
  pytest  --disable-warnings --input-path=<PATH_TO_INPUT_IMAGE> models/experimental/yolov4/demo/demo.py
  ```

Once you run the command, The output file named `ttnn_prediction_demo.jpg` will be generated.
