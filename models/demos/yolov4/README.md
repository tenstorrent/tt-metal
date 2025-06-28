# Yolov4

## Platforms:
    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```
To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Details

- The entry point to the yolov4 is located at:`models/demos/yolov4/tt/yolov4.py`
- Batch Size :1
- Supported Input Resolution - (640,640), (320,320) (Height,Width)


## How to run
Use the following command to run the model :

#### For 320x320:
```
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```
#### For 640x640:
```
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_true-0]
```

### Model performant running with Trace+2CQ

#### For 320x320:
- end-2-end perf is 114 FPS
  ```
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```
#### For 640x640:
- end-2-end perf is 56 FPS
  ```
  pytest models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution1-1-act_dtype0-weight_dtype0-device_params0]
  ```

### Single Image Demo

- Use the following command to run the yolov4 with a giraffe image:

For 320x320:
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4[device_params0-resolution0]
  ```
For 640x640:
  ```
  pytest models/demos/yolov4/demo.py::test_yolov4[device_params0-resolution1]
  ```
- Output images will be saved in the yolov4_predictions/ folder.

- To run the test with a different input image, add your image path to the `imgfile` parameter in the `@pytest.mark.parametrize` section of `demo.py`:

### Web Demo
- Try the interactive web demo (35 FPS end-2-end) for 320x320 following the [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/README.md)
