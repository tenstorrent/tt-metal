## YOLOv11n - Model

### Introduction

**YOLOv11** is the latest iteration in the YOLO series, offering improvements in accuracy, speed, and efficiency for real-time object detection. It features enhanced architecture and optimized training methods, suitable for various computer vision tasks.

### Model Details

* **Entry Point:** `models/experimental/yolov11/tt/ttnn_yolov11.py`
* **Weights:** `models/experimental/yolov11/reference/yolov11n.pt`

### Batch Size

* Default: 1
* Recommended: 1 for optimal performance

### Running YOLOv11 Demo

* **Single Image (640x640x3 or 224x224x3):** `pytest models/experimental/yolov11/demo/demo.py`
* **Dataset Evaluation:** `pytest models/experimental/yolov11/demo/evaluate.py`
  * Validation accuracy: 0.5616 on 250 images (coco-2017)

### Input and Output Data

* **Input Directory:** `models/experimental/yolov11/demo/images`
* **Output Directory:** `models/experimental/yolov11/demo/runs`
  * Torch model output: `torch_model`
  * TTNN model output: `tt_model`

### Pending Issues

* [#17385](https://github.com/tenstorrent/tt-metal/issues/17835) - Tracing fails in Yolov11n model
