# Yolov4 ttnn Demo

## Inputs required:

- Pretrained weights are loaded from weka path.
- If weka path is not available weights can be downloaded from [here](https://drive.google.com/file/d/1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ/view)  and place it to `tests/ttnn/integration_tests/yolov4/yolov4.pth`

## To run the demo:
1. Checkout to branch `ankit/yolov4_integration`
2. Run `pytest tests/ttnn/integration_tests/yolov4/test_ttnn_yolov4.py`
