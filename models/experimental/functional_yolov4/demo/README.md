# Yolov4 ttnn Demo

## Inputs required:


- A demo input image 320x320.
- coco.names which has class names.
- current branch  - `ankit/ttnn_yolov4_demo`.
- Input image and class names file available under `tests/ttnn/integration_tests/yolov4/`
- Pretrained weights are loaded from weka path.
- If weka path is not available weights can be downloaded from [here](https://drive.google.com/file/d/1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ/view)  and place it to `tests/ttnn/integration_tests/yolov4/yolov4.pth`.
## To run the demo:
1. Checkout to branch `ankit/ttnn_yolov4_demo`.
2. Run `pytest models/experimental/functional_yolov4/demo/demo.py`.
3. Prediction image will be generated at TT_METAL_HOME (current) directory.
