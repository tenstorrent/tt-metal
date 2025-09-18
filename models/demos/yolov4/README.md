# Yolov4

## Platforms:
    Wormhole (n150, n300)

## Introduction
YOLOv4 is a state-of-the-art real-time object detection model introduced in 2020 as an improved version of the YOLO (You Only Look Once) series. Designed for both speed and accuracy, YOLOv4 leverages advanced techniques such as weighted residual connections, cross-stage partial connections, and mosaic data

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
### For 320x320:
```
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```
### For 640x640:
```
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_true-0]
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- For `320x320`, end-2-end perf is `156` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
  ```
  models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution0-103-1-act_dtype0-weight_dtype0-device_params0]
  ```
- For `640x640`, end-2-end perf is `74` FPS  (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
  ```
  models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution1-46-1-act_dtype0-weight_dtype0-device_params0]
  ```

#### Multi Device (DP=2, N300):
- For `320x320`, end-2-end perf is `223` FPS
  ```
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant_dp[wormhole_b0-resolution0-103-1-act_dtype0-weight_dtype0-device_params0]
  ```
- For `640x640`, end-2-end perf is `113` FPS
  ```
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant_dp[wormhole_b0-resolution1-46-1-act_dtype0-weight_dtype0-device_params0]
  ```


### Demo
**Note:** Output images will be saved in the `yolov4_predictions/` folder.

#### Single Device (BS=1):
##### Custom Images:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4[resolution0-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4[resolution1-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```
- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov4/resources/` and run:
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4[resolution1-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```

##### Coco-2017 dataset:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco[resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```

#### Multi Device (DP=2, N300):
##### Custom Images:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_dp[wormhole_b0-resolution0-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_dp[wormhole_b0-resolution1-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```
- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov4/resources/` and run:
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_dp[wormhole_b0-resolution1-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```

##### Coco-2017 dataset:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco_dp[wormhole_b0-resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco_dp[wormhole_b0-resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```

#### Web Demo
- Try the interactive web demo (35 FPS end-2-end) for 320x320 following the [./web_demo/README.md](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/web_demo/README.md)

## Details
- The entry point to the `yolov4` is located at:`models/demos/yolov4/tt/yolov4.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(320, 320)`, `(640, 640)` - (Height, Width).
