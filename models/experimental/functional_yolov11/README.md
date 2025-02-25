## YOLOv11n - Model

#### Introduction

**YOLOv11** is the latest iteration in the YOLO series, bringing cutting-edge improvements in accuracy, speed, and efficiency for real-time object detection. Building on the success of previous versions, YOLOv11 introduces enhanced architecture and optimized training methods, making it a versatile solution for a wide range of computer vision tasks, from object detection to image classification and pose estimation.

#### Model Details

* The entry point to the YOLOv11 model is located in:
`models/experimental/functional_yolov11/tt/ttnn_yolov11.py`

* The model picks up weights from the **yolov11n.pt** file located in:
`models/experimental/functional_yolov11/reference/yolov11n.pt`

#### Batch Size:
* Set to 1 by default.
* Batch size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage.
* It's recommended to keep the batch size to **1** for optimal performance.

#### Running YOLOv11 Demo
* To run the YOLOv11 demo for different resolutions (**224x224** and **640x640**), use the following command:
`pytest --disable-warnings models/experimental/functional_yolov11/demo/demo.py`

#### Input Data
* By default, the demo will receive inputs from the `models/experimental/functional_yolov11/demo/images` directory. To test the model on different input data, simply add new image files to this directory.

#### Output Data

* The output from the model will be saved in a **runs** folder created inside:
`models/experimental/functional_yolov11/demo/`
* For reference:
The model output(torch model) will be stored in the **torch_model** directory.
The TTNN model output will be stored in the **tt_model** directory.

#### Pending Issues:
* [#17385](https://github.com/tenstorrent/tt-metal/issues/17835) - Tracing fails in Yolov11n model
