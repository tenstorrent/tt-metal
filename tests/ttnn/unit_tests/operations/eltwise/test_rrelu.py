# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),  # default values
        (0.0, 0.5),
        (0.1, 0.3),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
    ],
)
def test_rrelu(shape, lower, upper, dtype, device):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)

    # PyTorch reference: rrelu in eval mode = leaky_relu with slope = (lower + upper) / 2
    slope = (lower + upper) / 2.0
    torch_output = torch.where(torch_input >= 0, torch_input, slope * torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Use PCC for comparison; bfloat16 has lower precision
    passing, pcc_msg = assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
    assert passing, f"PCC check failed: {pcc_msg}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_rrelu_default_params(shape, device):
    """Test rrelu with default parameters (lower=0.125, upper=1/3)."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    slope = (0.125 + 1.0 / 3.0) / 2.0
    torch_output = torch.where(torch_input >= 0, torch_input, slope * torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    passing, pcc_msg = assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
    assert passing, f"PCC check failed: {pcc_msg}"
