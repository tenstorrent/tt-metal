## functional_mobilenetv2 Demo
## How to Run

Download the pre-trained model from https://download.pytorch.org/models/mobilenet_v2-b0353104.pth and move the file to tests/ttnn/integration_tests/mobilenetv2.

Use `pytest tests/ttnn/integration_tests/mobilenetv2/test_tttnn_mobilenetv2.py` to run the demo.

## Details

The entry point to functional_mobilenetv2 model is TtMobilenetv2 in `models/experimental/functional_mobilenetv2/tt/ttnn_mobilenetv2.py` for ttnn implementation.
(`models/experimental/functional_mobilenetv2/reference/mobilenetv2.py` for reference).

## Issues

Since TTNN doesn't support conv with groups>1, the implementation of mobilenetv2 in TTNN will result with the following error.
`AssertionError: Only convs with groups == 1 supported`
