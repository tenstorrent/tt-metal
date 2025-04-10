## YOLOv11n - Model

### Introduction

**YOLOv11n** is the smallest variant in the YOLOV11 series, it offers improvements in accuracy, speed, and efficiency for real-time object detection. It features enhanced architecture and optimized training methods, suitable for various computer vision tasks.

### Model Details

* **Entry Point:** `models/experimental/yolov11/tt/ttnn_yolov11.py`

### Running YOLOv11

* **Single Image (640x640x3 or 224x224x3):** `pytest --disable-warnings tests/ttnn/integration_tests/yolov11/test_ttnn_yolo_v11.py`

### Pending Issues

* [#17385](https://github.com/tenstorrent/tt-metal/issues/17835) - Tracing fails in Yolov11n model
