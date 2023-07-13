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
from python_api_testing.models.EfficientNet.tt.efficientnet_mbconv import (
    TtEfficientnetMbConv,
    MBConvConfig,
)


def test_efficientnet_mbconv():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()
    refence_module = refence_model.features[1][0]

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)
    pt_out = refence_module(test_input)

    mb_conv_config = MBConvConfig(
        expand_ratio=1,
        kernel=3,
        stride=1,
        input_channels=32,
        out_channels=16,
        num_layers=1,
    )

    tt_module = TtEfficientnetMbConv(
        state_dict=refence_model.state_dict(),
        base_address=f"features.1.0",
        device=device,
        cnf=mb_conv_config,
        stochastic_depth_prob=0.0,
    )

    test_input = torch2tt_tensor(
        test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
    )

    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_mbconv Passed!")
    else:
        logger.warning("test_efficientnet_mbconv Failed!")

    assert does_pass
