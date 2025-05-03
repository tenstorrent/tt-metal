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
from models.experimental.efficientnet.tt.efficientnet_fused_mbconv import (
    TtEfficientnetFusedMBConv,
    FusedMBConvConfig,
)


def test_efficientnet_fused_mbconv(device):
    refence_model = torchvision.models.efficientnet_v2_s(pretrained=True)
    refence_model.eval()
    refence_module = refence_model.features[2][0]

    torch.manual_seed(0)
    test_input = torch.rand(1, 24, 64, 64)
    pt_out = refence_module(test_input)

    mb_conv_config = FusedMBConvConfig(
        expand_ratio=4,
        kernel=3,
        stride=2,
        input_channels=24,
        out_channels=48,
        num_layers=4,
    )

    tt_module = TtEfficientnetFusedMBConv(
        state_dict=refence_model.state_dict(),
        base_address=f"features.2.0",
        device=device,
        cnf=mb_conv_config,
        stochastic_depth_prob=0.0,
    )

    test_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_efficientnet_fused_mbconv Passed!")
    else:
        logger.warning("test_efficientnet_fused_mbconv Failed!")

    assert does_pass
