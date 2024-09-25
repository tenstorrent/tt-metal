# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


def run_moe_test(N, C, H, W, k, E, e, dtype, device):
    # torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input[:, :, :, E:] = 0  # padded input

    expert_mask = torch.zeros([N, C, 1, W], dtype=torch_dtype)
    expert_mask[:, :, :, E:] = float("-inf")

    # TODO: make this addition a part of the moe op
    input += expert_mask

    topE_mask = torch.zeros([N, C, 1, k], dtype=torch_dtype)
    topE_mask[:, :, :, e:] = float("-inf")

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=-1)
    torch_weights_1SB1 = torch.sum(
        (torch.softmax(pyt_topk_values + topE_mask, dim=-1) * (pyt_topk_indices == 0))[:, :, :, :e],
        dim=-1,
        keepdim=True,
    )
    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_expert_mask = ttnn.from_torch(expert_mask, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_topE_mask = ttnn.from_torch(topE_mask, dtype, layout=ttnn.Layout.TILE, device=device)

    for i in range(3):
        weights_1SB1 = ttnn.moe(ttnn_input, ttnn_expert_mask, ttnn_topE_mask, k)

        assert list(weights_1SB1.shape.with_tile_padding()) == [N, C, H, k]

        ttnn_weights_1SB1 = ttnn.to_torch(weights_1SB1)

        pcc_values = 0.95

        assert_with_pcc(torch_weights_1SB1, ttnn_weights_1SB1, pcc_values)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "BFLOAT16_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, k, E, e",
    ((1, 1, 32, 64, 32, 8, 2),),  # Mixtral8x7B
)
def test_moe(N, C, H, W, k, E, e, dtype, device, use_program_cache):
    run_moe_test(N, C, H, W, k, E, e, dtype, device)
