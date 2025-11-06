# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger
from models.common.utility_functions import tt_to_torch_tensor
from models.experimental.SSD512.tt.layers.tt_l2norm import TtL2Norm
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_l2norm(device):
    """Test L2Norm implementation"""
    # set parameters for the test
    n_channels = 512
    scale = 20.0
    shape = (2, n_channels, 32, 32)
    torch_input = torch.randn(shape)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Create L2Norm layers
    torch_l2norm = nn.Sequential(
        nn.BatchNorm2d(n_channels, affine=False),
        nn.Conv2d(n_channels, n_channels, 1, groups=n_channels, bias=False),
    )
    # Initialize scale weights
    torch_l2norm[1].weight.data.fill_(scale)

    # TTNN L2Norm
    tt_l2norm = TtL2Norm(n_channels, scale=scale, device=device)

    # Run forward pass
    torch_output = torch_l2norm(torch_input)
    tt_output = tt_l2norm(ttnn_input)
    tt_output_torch = tt_to_torch_tensor(tt_output)
    ttnn_output = torch.permute(tt_output_torch, (0, 3, 1, 2))

    output_pass, pcc_value = comp_pcc(torch_output, tt_output_torch, 0.99)
    logger.info(f"L2Norm PCC: {pcc_value}")

    assert_with_pcc(torch_output, tt_output_torch, 0.99)
