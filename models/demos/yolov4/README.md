# Yolov4

### Platforms:
    WH - N150, N300

### Note:

- On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

- Or, make sure to set the following environment variable in the terminal:
  ```
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  ```
- To obtain the perf reports through profiler, please build with following command:
  ```
  ./build_metal.sh -p
  ```

## Details

- The entry point to the `yolov4` is located at:`models/demos/yolov4/tt/yolov4.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(320, 320)`, `(640, 640)` - (Height, Width).


## How to run
Use the following command(s) to run the model :

#### For 320x320:
  ```
  pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
  ```
#### For 640x640:
  ```
  pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_true-0]
  ```

## Model performant running with Trace+2CQ

### Single Device (BS=1):

- For `320x320`, end-2-end perf is `114` FPS
  ```
  models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution0-103-1-act_dtype0-weight_dtype0-device_params0]
  ```

- For `640x640`, end-2-end perf is `56` FPS
  ```
  models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution1-46-1-act_dtype0-weight_dtype0-device_params0]
  ```

### Multi Device (DP=2, N300):

- For `320x320`, end-2-end perf is `224.56` FPS
  ```
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant_dp[wormhole_b0-resolution0-103-1-act_dtype0-weight_dtype0-device_params0]
  ```

- For `640x640`, end-2-end perf is `93.17` FPS
  ```
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant_dp[wormhole_b0-resolution1-46-1-act_dtype0-weight_dtype0-device_params0]
  ```


## Demo

#### Note: Output images will be saved in the `yolov4_predictions/` folder.

### Single Device (BS=1):

#### Custom Images:

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

#### Coco-2017 dataset:

- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco[resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```

### Multi Device (DP=2, N300):

#### Custom Images:

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

#### Coco-2017 dataset:

- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco_dp[wormhole_b0-resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4_coco_dp[wormhole_b0-resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```


### Web Demo
- Try the interactive web demo (35 FPS end-2-end) for 320x320 following the [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/README.md)
