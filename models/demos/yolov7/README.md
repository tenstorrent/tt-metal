# Yolov7

## Platforms:
    Wormhole (n150, n300)

## Introduction
YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy. It builds on the YOLO family of detectors and introduces significant architectural improvements for enhanced speed and accuracy. YOLOv7 supports advanced features such as model reparameterization, extended model scaling, and multi-task capabilities including object detection, instance segmentation, and pose estimation.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Install Packages before running the tests if not present:
- ``` sudo apt-get update && sudo apt-get install -y graphviz ```

Ideally, ultralytics should be automatically installed while running ./create_venv.sh but if still package error then manually follow the below commands
```
pip3 install ultralytics
or
python_env/bin/python -m ensurepip --upgrade      (if trying to install inside the venv)
python_env/bin/python -m pip install ultralytics
```

## How to Run
- Use the following command to run the yolov7 model
```python
pytest --disable-warnings models/demos/yolov7/tests/pcc/test_ttnn_yolov7.py
```
**Current PCC Results:** PCC (threshold: 0.999): 0.9991836

### Model Performant with Trace+2CQ
#### Single Device (BS=1):
- For `640x640`, end-2-end perf is `120` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
  ```bash
  pytest --disable-warnings models/demos/yolov7/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```
  _Note: Original FPS: 123_

#### Multi Device (DP=2, N300):
- For `640x640`, end-2-end perf is `199.9` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov7/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```
  _Note: Original FPS: 214_

### Demo

### Single Device (BS=1)
#### Custom Images:
- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo
    ```
- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov7/demo/images`

#### Coco-2017 dataset:
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300)
#### Custom Images:
- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo_dp
  ```

#### Coco-2017 dataset:
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo_dataset_dp
  ```

Note: Output images will be saved in the `models/demos/yolov7/demo/runs` folder.

## Testing

### Performant evaluation with Trace+2CQ
Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest models/demos/yolo_eval/evaluate.py::test_yolov7[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.



## Details
- The entry point of the model is located at ```models/demos/yolov7/tt/ttnn_yolov7.py```
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
