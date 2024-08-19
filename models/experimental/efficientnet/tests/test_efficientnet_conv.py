# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import torchvision

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from models.experimental.efficientnet.tt.efficientnet_conv import (
    TtEfficientnetConv2d,
    TtEfficientnetConv2dNormActivation,
)
from models.experimental.efficientnet.tt.efficientnet_model import reference_efficientnet_lite0


def run_efficientnet_conv2d(state_dict, base_address, reference_module, device):
    in_channels = reference_module.in_channels
    out_channels = reference_module.out_channels
    kernel_size = reference_module.kernel_size
    stride = reference_module.stride
    padding = reference_module.padding
    groups = reference_module.groups
    dilation = reference_module.dilation

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 224, 224)
    pt_out = reference_module(test_input)

    tt_module = TtEfficientnetConv2d(
        state_dict=state_dict,
        base_address=base_address,
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

    test_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_conv2d Passed!")
    else:
        logger.warning("test_efficientnet_conv2d Failed!")

    assert does_pass


def test_efficientnet_conv2d_b0(device):
    reference_model = torchvision.models.efficientnet_b0(pretrained=True)
    reference_model.eval()

    run_efficientnet_conv2d(
        state_dict=reference_model.state_dict(),
        base_address=f"features.0.0",
        reference_module=reference_model.features[0][0],
        device=device,
    )


def test_efficientnet_conv2d_lite0(device):
    reference_model = reference_efficientnet_lite0()

    run_efficientnet_conv2d(
        state_dict=reference_model.state_dict(),
        base_address=f"stem.0",
        reference_module=reference_model.stem[0],
        device=device,
    )


def run_efficientnet_conv_norm_activation(
    device, state_dict, conv_base_address, bn_base_address, reference_module, is_lite
):
    in_channels = reference_module[0].in_channels
    out_channels = reference_module[0].out_channels
    kernel_size = reference_module[0].kernel_size
    stride = reference_module[0].stride
    padding = reference_module[0].padding
    groups = reference_module[0].groups
    dilation = reference_module[0].dilation
    # activation_layer = reference_module.activation_layer

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
    pt_out = reference_module(test_input)

    tt_module = TtEfficientnetConv2dNormActivation(
        state_dict=state_dict,
        conv_base_address=conv_base_address,
        bn_base_address=bn_base_address,
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
        is_lite=is_lite,
    )

    test_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_conv_norm_activation Passed!")
    else:
        logger.warning("test_efficientnet_conv_norm_activation Failed!")

    assert does_pass


def test_efficientnet_conv_norm_activation_b0(device):
    reference_model = torchvision.models.efficientnet_b0(pretrained=True)
    reference_model.eval()

    run_efficientnet_conv_norm_activation(
        device,
        state_dict=reference_model.state_dict(),
        conv_base_address=f"features.0.0",
        bn_base_address=f"features.0.1",
        reference_module=reference_model.features[0],
        is_lite=False,
    )


def test_efficientnet_conv_norm_activation_lite0(device):
    reference_model = reference_efficientnet_lite0()

    run_efficientnet_conv_norm_activation(
        device,
        state_dict=reference_model.state_dict(),
        conv_base_address=f"stem.0",
        bn_base_address=f"stem.1",
        reference_module=reference_model.stem,
        is_lite=True,
    )
