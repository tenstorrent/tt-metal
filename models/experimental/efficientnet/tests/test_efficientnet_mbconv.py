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
from models.experimental.efficientnet.tt.efficientnet_mbconv import (
    TtEfficientnetMbConv,
    MBConvConfig,
)
from models.experimental.efficientnet.tt.efficientnet_model import reference_efficientnet_lite0


def run_efficientnet_mbconv(device, state_dict, base_address, reference_module, mb_conv_config, is_lite):
    torch.manual_seed(0)
    test_input = torch.rand(1, mb_conv_config.input_channels, 64, 64)
    pt_out = reference_module(test_input)

    tt_module = TtEfficientnetMbConv(
        state_dict=state_dict,
        base_address=base_address,
        device=device,
        cnf=mb_conv_config,
        stochastic_depth_prob=0.0,
        is_lite=is_lite,
    )

    test_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_mbconv Passed!")
    else:
        logger.warning("test_efficientnet_mbconv Failed!")

    assert does_pass


def test_efficientnet_mbconv_b0(device):
    reference_model = torchvision.models.efficientnet_b0(pretrained=True)
    reference_model.eval()

    mb_conv_config = MBConvConfig(
        expand_ratio=1,
        kernel=3,
        stride=1,
        input_channels=32,
        out_channels=16,
        num_layers=1,
    )

    run_efficientnet_mbconv(
        device,
        state_dict=reference_model.state_dict(),
        base_address=f"features.1.0",
        reference_module=reference_model.features[1][0],
        mb_conv_config=mb_conv_config,
        is_lite=False,
    )


def test_efficientnet_mbconv_lite0(device):
    reference_model = reference_efficientnet_lite0()

    mb_conv_config = MBConvConfig(
        expand_ratio=1,
        kernel=3,
        stride=1,
        input_channels=32,
        out_channels=16,
        num_layers=1,
    )

    run_efficientnet_mbconv(
        device,
        state_dict=reference_model.state_dict(),
        base_address=f"blocks.0.0",
        reference_module=reference_model.blocks[0][0],
        is_lite=True,
        mb_conv_config=mb_conv_config,
    )


def test_efficientnet_mbconv_v2_s(device):
    reference_model = torchvision.models.efficientnet_v2_s(pretrained=True)
    reference_model.eval()

    mb_conv_config = MBConvConfig(
        expand_ratio=4,
        kernel=3,
        stride=2,
        input_channels=64,
        out_channels=128,
        num_layers=6,
    )

    run_efficientnet_mbconv(
        device,
        state_dict=reference_model.state_dict(),
        base_address=f"features.4.0",
        reference_module=reference_model.features[4][0],
        is_lite=False,
        mb_conv_config=mb_conv_config,
    )
