# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.groupnorm import GroupNorm1D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_length,channels,num_groups",
    [
        (2, 64, 32, 32),
        (1, 512, 128, 128),
        (1, 1024, 256, 256),
        (1, 1024, 512, 512),
        (1, 2048, 768, 768),
        (1, 113986, 512, 512),
    ],
    ids=[
        "b2_l64_c32_g32",
        "b1_l512_c128_g128",
        "b1_l1024_c256_g256",
        "b1_l1024_c512_g512",
        "b1_l2048_c768_g768",
        "b1_l113986_c512_g512",
    ],
)
def test_groupnorm(device, batch_size, input_length, channels, num_groups):
    torch.manual_seed(0)

    torch_groupnorm = torch.nn.GroupNorm(num_groups, channels, eps=1e-5, affine=True).eval()

    torch_input = torch.randn(batch_size, input_length, channels, dtype=torch.float32)
    torch_output = torch_groupnorm(torch_input.permute(0, 2, 1)).permute(0, 2, 1)

    tt_groupnorm = GroupNorm1D(
        device=device,
        num_channels=channels,
        num_groups=num_groups,
        # dtype=ttnn.bfloat16,
    )
    parameters = {
        "norm.groupnorm.weight": torch_groupnorm.weight,
        "norm.groupnorm.bias": torch_groupnorm.bias,
    }
    tt_groupnorm.load_parameters(parameters=parameters, key="groupnorm", prefix="norm.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output = tt_groupnorm.gp_slice(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
