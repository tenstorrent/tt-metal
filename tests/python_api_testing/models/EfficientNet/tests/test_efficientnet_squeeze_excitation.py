import os
import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import tt_lib
import torch
from loguru import logger
import torchvision

from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from python_api_testing.models.EfficientNet.tt.efficientnet_squeeze_excitation import (
    TtEfficientnetSqueezeExcitation,
)


def test_efficientnet_squeeze_excitation():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()

    block = 1
    refence_module = refence_model.features[block][0].block[1]

    input_channels = refence_module.fc1.in_channels
    squeeze_channels = refence_module.fc1.out_channels

    logger.debug(f"input_channels {input_channels}")
    logger.debug(f"squeeze_channels {squeeze_channels}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)
    pt_out = refence_module(test_input)

    tt_module = TtEfficientnetSqueezeExcitation(
        state_dict=refence_model.state_dict(),
        base_address=f"features.{block}.0.block.1",
        device=device,
        input_channels=input_channels,
        squeeze_channels=squeeze_channels,
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
        logger.info("test_efficientnet_squeeze_excitation Passed!")
    else:
        logger.warning("test_efficientnet_squeeze_excitation Failed!")

    assert does_pass
