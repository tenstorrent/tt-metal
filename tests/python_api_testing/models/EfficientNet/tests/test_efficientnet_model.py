import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import tt_lib
import torch
from loguru import logger
import torchvision

from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from python_api_testing.models.EfficientNet.tt.efficientnet_model import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)


def run_efficientnet_model_test(reference_model_class, tt_model_class):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = reference_model_class(pretrained=True)
    refence_model.eval()

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 224, 224)
    pt_out = refence_model(test_input)

    tt_model = tt_model_class(device)

    test_input = torch2tt_tensor(
        test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
    )

    tt_out = tt_model(test_input)
    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_efficientnet_model {tt_model_class} Passed!")
    else:
        logger.warning(f"test_efficientnet_model {tt_model_class} Failed!")

    assert does_pass


def test_efficientnet_b0_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b0, efficientnet_b0)


def test_efficientnet_b1_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b1, efficientnet_b1)


def test_efficientnet_b2_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b2, efficientnet_b2)


def test_efficientnet_b3_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b3, efficientnet_b3)


def test_efficientnet_b4_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b4, efficientnet_b4)


def test_efficientnet_b5_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b5, efficientnet_b5)


def test_efficientnet_b6_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b6, efficientnet_b6)


def test_efficientnet_b7_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b7, efficientnet_b7)
