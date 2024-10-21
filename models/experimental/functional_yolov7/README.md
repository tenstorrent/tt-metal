## Yolov7 Model

# Platforms:
    WH N300, N150

## command to run the model
Use the following command to run the yolov7 model
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov7/test_ttnn_yolov7.py
```

## Demo
Batch size : 1
Use the following command to run the demo
```
pytest --disable-warnings models/experimental/functional_yolov7/demo/yolov7_demo.py
```

## Outputs
The Demo outputs are saved inside this directory: `models/experimental/functional_yolov7/demo/runs/detect`

**Note:** Current demo results are not as accurate as torch, yet to improve demo results further.

Torch ops: mul, pow, add, sub (Overall PCC of the model drops when converted to ttnn)

Pending issues:
[#17583](https://github.com/tenstorrent/tt-metal/issues/17583) - Yolov7 Trace+2cq fails with Out of Memoy issue
[#19256](https://github.com/tenstorrent/tt-metal/issues/19256) - Yolov7 Demo gives poor results
