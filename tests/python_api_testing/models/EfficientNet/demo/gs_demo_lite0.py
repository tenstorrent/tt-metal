from tests.python_api_testing.models.EfficientNet.demo.gs_demo_b0 import run_gs_demo
from tests.python_api_testing.models.EfficientNet.tt.efficientnet_model import (
    efficientnet_lite0,
)


def test_gs_demo_lite0(imagenet_label_dict):
    run_gs_demo(efficientnet_lite0, imagenet_label_dict)
