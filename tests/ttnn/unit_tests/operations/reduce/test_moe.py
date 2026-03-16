# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.reduce.numeric_check import (
    collect_and_dump_numeric_metrics,
)


def run_moe_test(N, C, H, W, k, E, e, dtype, device):
    # torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16
    torch.manual_seed(2005)
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

        assert list(weights_1SB1.padded_shape) == [N, C, H, k]

        ttnn_weights_1SB1 = ttnn.to_torch(weights_1SB1)

        pcc_values = 0.95
        # Collect numeric metrics and dump to CSV using reusable function
        test_name = f"test_moe[N={N},C={C},H={H},W={W},k={k},E={E},e={e},dtype={dtype},iteration={i}]"
        collect_and_dump_numeric_metrics(
            torch_weights_1SB1,
            ttnn_weights_1SB1,
            test_name=test_name,
            csv_filename="test_moe_numeric_results.csv",
            test_params=None,
        )
        assert_with_pcc(torch_weights_1SB1, ttnn_weights_1SB1, pcc_values)


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
def test_moe(N, C, H, W, k, E, e, dtype, device):
    run_moe_test(N, C, H, W, k, E, e, dtype, device)
