# Yolov5x - Model

### Platforms:

Wormhole N150, N300

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction

**YOLOv5x** is the largest variant in the YOLOv5 series, delivering top-tier accuracy and performance for advanced object detection tasks. It features a deeper and wider architecture, making it ideal for high-precision applications where accuracy is prioritized over model size or inference speed.


### Details

- The entry point to the yolov11 is located at:`models/experimental/yolov5x/tt/yolov5x.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings models/experimental/yolov5x/tests/pcc/test_ttnn_yolov5x.py::test_yolov5x
```
