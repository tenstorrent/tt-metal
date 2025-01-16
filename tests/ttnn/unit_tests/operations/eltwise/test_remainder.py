# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_broken_remainder(input_shapes, device):
    torch_lhs = torch.ones(32, 32, dtype=torch.bfloat16)
    torch_rhs = torch.zeros(32, 32, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_lhs, torch_rhs, device=device)

    tt_lhs = ttnn.from_torch(torch_lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_rhs = ttnn.from_torch(torch_rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_result = ttnn.remainder(tt_lhs, tt_rhs)
    result = ttnn.to_torch(tt_result)
    assert torch.allclose(result, golden, atol=0.01, rtol=0)


@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_broken_remainder1(input_shapes, device):
    torch_lhs = torch.ones(32, 32, dtype=torch.bfloat16) * 95
    torch_rhs = torch.ones(32, 32, dtype=torch.bfloat16) * (-94.5)

    golden_function = ttnn.get_golden_function(ttnn.remainder)  # all -94.0
    golden = golden_function(torch_lhs, torch_rhs, device=device)

    tt_lhs = ttnn.from_torch(torch_lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_rhs = ttnn.from_torch(torch_rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    tt_result = ttnn.remainder(tt_lhs, tt_rhs)
    result = ttnn.to_torch(tt_result)  # all 0.5
    assert torch.allclose(result, golden, atol=0.01, rtol=0)
