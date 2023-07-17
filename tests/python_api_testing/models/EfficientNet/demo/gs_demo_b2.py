import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.EfficientNet.demo.gs_demo_b0 import run_gs_demo
from python_api_testing.models.EfficientNet.tt.efficientnet_model import efficientnet_b2


def test_gs_demo_b2(imagenet_label_dict):
    run_gs_demo(efficientnet_b2, imagenet_label_dict)
