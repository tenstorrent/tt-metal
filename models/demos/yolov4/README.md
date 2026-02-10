# Yolov4

## Platforms:
    Wormhole (n150, n300)

## Introduction
YOLOv4 is a state-of-the-art real-time object detection model introduced in 2020 as an improved version of the YOLO (You Only Look Once) series. Designed for both speed and accuracy, YOLOv4 leverages advanced techniques such as weighted residual connections, cross-stage partial connections, and mosaic data

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

## Download Model Weights

Before running perf tests, download the YOLOv4 model weights file. The perf tests require `models/demos/yolov4/tests/pcc/yolov4.pth` to be present.

### Automatic Download (Recommended):

Run the download script from the repository root:

```bash
bash models/demos/yolov4/tests/pcc/yolov4_weights_download.sh
```

**Note:** This script requires `gdown` to be installed. If not installed, it will attempt to install it automatically. If the automatic download fails, use the manual method below.

### Manual Download:

If the automatic download script fails, you can manually download the weights:

1. Install `gdown` if not already installed:
   ```bash
   pip3 install gdown
   or
   python_env/bin/python -m pip install gdown (if trying to install inside the venv)
   ```

2. Run the Download script:
   ```bash models/demos/yolov4/tests/pcc/yolov4_weights_download.sh
    ```

3. Verify the file exists:
   ```bash
   ls -lh models/demos/yolov4/tests/pcc/yolov4.pth
   ```

**Troubleshooting:** If you encounter `FileNotFoundError: [Errno 2] No such file or directory: 'models/demos/yolov4/tests/pcc/yolov4.pth'` when running perf tests, ensure:
- You have downloaded the weights file using one of the methods above
- You are running pytest from the repository root directory
- The file path `models/demos/yolov4/tests/pcc/yolov4.pth` exists relative to your current working directory

## How to Run
### For 320x320:
```
pytest --disable-warnings models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
pytest --disable-warnings models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_false-0]
```
### For 640x640:
```
pytest --disable-warnings models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_true-0]
pytest --disable-warnings models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_false-0]
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- For `320x320`, end-2-end perf is `166` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
  ```
  pytest --disable-warnings models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution0-166-1-DataType.BFLOAT8_B-DataType.BFLOAT8_B-device_params0]
  ```
- For `640x640`, end-2-end perf is `74` FPS  (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
  ```
  pytest --disable-warnings models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution1-74-1-DataType.BFLOAT8_B-DataType.BFLOAT8_B-device_params0]
  ```

#### Multi Device (DP=2, N300):
- For `320x320`, end-2-end perf is `254` FPS
  ```
  pytest --disable-warnings models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant_dp[wormhole_b0-resolution0-254-1-DataType.BFLOAT16-DataType.BFLOAT16-device_params0]
  ```
- For `640x640`, end-2-end perf is `123` FPS
  ```
  pytest --disable-warnings models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant_dp[wormhole_b0-resolution1-123-1-DataType.BFLOAT16-DataType.BFLOAT16-device_params0]
  ```

## Current Model Performance Summary

**Note:** Performance numbers are measured on **N150 AND N300** platform.

| Resolution | Pretrained Weights | Boxes PCC (threshold: 0.99) | Confs PCC (threshold: 0.9) | Performance (FPS, N150)  | Demo Status |
|------------|--------------------|-----------------------------|----------------------------|--------------------------|-------------|
| 640x640     | False             | 0.9999884                   | 0.9937709                  | 86.7                     | Passed      |
| 640x640     | True              | 0.9991520                   | 0.9368130                  | 86.7                     | Passed      |
| 320x320     | False             | 0.9999879                   | 0.9937709                  | 184                      | Passed      |
| 320x320     | True              | 0.9976081                   | 0.9537761                  | 184                      | Passed      |

| Resolution | Pretrained Weights | Boxes PCC (threshold: 0.99) | Confs PCC (threshold: 0.9) | Performance (FPS, N300)  | Demo Status |
|------------|--------------------|-----------------------------|----------------------------|--------------------------|-------------|
| 640x640    | False              | 0.9999884                   | 0.9937709                  | 136                      | Passed      |
| 640x640    | True               | 0.9991520                   | 0.9368130                  | 136                      | Passed      |
| 320x320    | False              | 0.9999879                   | 0.9937709                  | 254                      | Passed      |
| 320x320    | True               | 0.9976081                   | 0.9537761                  | 254                      | Passed      |

### Demo
**Note:** Output images will be saved in the `yolov4_predictions/` folder.

#### Single Device (BS=1):
##### Custom Images:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4[resolution0-1-DataType.BFLOAT16-DataType.BFLOAT16-models/demos/yolov4/resources-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4[resolution1-1-DataType.BFLOAT16-DataType.BFLOAT16-models/demos/yolov4/resources-device_params0]
  ```
- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov4/resources/` and run:
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4[resolution1-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```

##### Coco-2017 dataset:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_coco[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_coco[resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```

#### Multi Device (DP=2, N300):
##### Custom Images:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_dp[wormhole_b0-resolution0-1-DataType.BFLOAT16-DataType.BFLOAT16-models/demos/yolov4/resources-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_dp[wormhole_b0-resolution1-1-DataType.BFLOAT16-DataType.BFLOAT16-models/demos/yolov4/resources-device_params0]
  ```
- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov4/resources/` and run:
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_dp[wormhole_b0-resolution1-1-act_dtype0-weight_dtype0-models/demos/yolov4/resources-device_params0]
  ```

##### Coco-2017 dataset:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_coco_dp[wormhole_b0-resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov4/demo.py::test_yolov4_coco_dp[wormhole_b0-resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```


#### Web Demo
- Try the interactive web demo (35 FPS end-2-end) for 320x320 following the [./web_demo/README.md](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/web_demo/README.md)

## Details
- The entry point to the `yolov4` is located at:`models/demos/yolov4/tt/yolov4.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(320, 320)`, `(640, 640)` - (Height, Width).
