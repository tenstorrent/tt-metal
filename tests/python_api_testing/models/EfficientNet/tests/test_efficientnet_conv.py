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
from python_api_testing.models.EfficientNet.tt.efficientnet_conv import (
    TtEfficientnetConv2d,
    TtEfficientnetConv2dNormActivation,
)


def test_efficientnet_conv2d():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()

    block = 0
    refence_module = refence_model.features[block][0]

    in_channels = refence_module.in_channels
    out_channels = refence_module.out_channels
    kernel_size = refence_module.kernel_size
    stride = refence_module.stride
    padding = refence_module.padding
    groups = refence_module.groups
    dilation = refence_module.dilation

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 224, 224)
    pt_out = refence_module(test_input)

    tt_module = TtEfficientnetConv2d(
        state_dict=refence_model.state_dict(),
        base_address=f"features.{block}.0",
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        conv_on_device=False,
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
        logger.info("test_efficientnet_conv2d Passed!")
    else:
        logger.warning("test_efficientnet_conv2d Failed!")

    assert does_pass


def test_efficientnet_conv_norm_activation():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()

    block = 0
    refence_module = refence_model.features[block]

    in_channels = refence_module[0].in_channels
    out_channels = refence_module[0].out_channels
    kernel_size = refence_module[0].kernel_size
    stride = refence_module[0].stride
    padding = refence_module[0].padding
    groups = refence_module[0].groups
    dilation = refence_module[0].dilation
    # activation_layer = refence_module.activation_layer

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")
    # logger.debug(f"act {activation_layer}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 224, 224)
    pt_out = refence_module(test_input)

    tt_module = TtEfficientnetConv2dNormActivation(
        state_dict=refence_model.state_dict(),
        base_address=f"features.{block}",
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        activation_layer=True,
        dilation=dilation,
        conv_on_device=False,
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
        logger.info("test_efficientnet_conv_norm_activation Passed!")
    else:
        logger.warning("test_efficientnet_conv_norm_activation Failed!")

    assert does_pass
