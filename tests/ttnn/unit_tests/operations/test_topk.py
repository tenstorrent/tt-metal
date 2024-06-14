# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_grayskull


def run_topk_test(N, C, H, W, k, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=-1, largest=True, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True)

    assert list(ttnn_topk_values.get_legacy_shape()) == [N, C, H, k]
    assert list(ttnn_topk_indices.get_legacy_shape()) == [N, C, H, k]

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
        pcc_index = 0.99
    else:
        pcc_index = 1.0
        pcc_values = 1.0

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # so we will use pcc for the values and not the indices
    # to make sure the indices are correct, we gather the relevant values from the original torch tensor and test to see if they are similar
    # rounding may also cause more ties than expected
    ttnn_torch_gather_from_indices = torch.gather(input, -1, ttnn_torch_indices.to(torch.int64))

    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)
    assert_with_pcc(pyt_topk_values, ttnn_torch_gather_from_indices, pcc_index)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        # ttnn.float32, top bits in float32 get cut off somewhere, LLK does not work for this
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        # "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, k,",
    (
        (1, 1, 32, 64, 32),
        (1, 1, 32, 256, 32),
        (1, 1, 128, 64, 32),
        (1, 1, 1024, 64, 32),
        (1, 1, 2048, 64, 32),
    ),
)
def test_topk(N, C, H, W, k, dtype, device):
    run_topk_test(N, C, H, W, k, dtype, device)
