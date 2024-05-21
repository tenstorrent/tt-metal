# Yolox_m Demo

## Inputs required:


- A demo input image
- current branch  - `ankit/ttnn_yolox_m`.
- Input image is available under `tests/ttnn/integration_tests/yolox_m/`
- Pretrained weights are loaded from weka path.
- If weka path is not available weights can be downloaded from [here](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)  and place it to `tests/ttnn/integration_tests/yolox_m/yolox_m.pth`.
## To run the ttnn demo:
1. Checkout to branch `ankit/ttnn_yolox_m`.
2. Run `pytest models/experimental/functional_yolox_m/demo/gs_demo.py `.
3. Prediction image will be generated at TT_METAL_HOME (current) directory.

## To run the torch reference demo:
1. Checkout to branch `ankit/ttnn_yolox_m`.
2. Run `pytest models/experimental/functional_yolox_m/demo/cpu_demo.py  `.
3. Prediction image will be generated at TT_METAL_HOME (current) directory.
