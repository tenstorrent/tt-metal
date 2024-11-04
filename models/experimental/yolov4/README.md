# Yolov4 Demo


## Inference basic demo
- Use the following command to run the yolov4 with giraffe image from weka path:
  ```python
  pytest  --disable-warnings models/experimental/yolov4/demo/demo.py
  ```

- Use the following command to run the yolov4 with different input image:
  ```python
  pytest  --disable-warnings --input-path=<PATH_TO_INPUT_IMAGE> models/experimental/yolov4/demo/demo.py
  ```

- Once you run the command, The output file named `ttnn_prediction_demo.jpg` will be generated.



## Performance demo (End 2 End)
```python
pytest  --disable-warnings models/experimental/yolov4/demo/demo_yolov4_2cq_trace.py
```
