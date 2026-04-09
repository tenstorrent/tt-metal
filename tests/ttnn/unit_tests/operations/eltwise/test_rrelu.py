# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ],
)
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),
        (0.0, 0.5),
        (0.1, 0.3),
    ],
)
def test_rrelu_eval_bfloat16(input_shape, lower, upper, device):
    """Test rrelu in eval mode (training=False) with bfloat16."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    golden = torch.nn.functional.rrelu(torch_input.float(), lower=lower, upper=upper, training=False).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    tt_result = ttnn.to_torch(tt_output)

    assert_with_ulp(golden, tt_result, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 3, 320, 384]),
    ],
)
def test_rrelu_eval_default_params(input_shape, device):
    """Test rrelu eval mode with default lower/upper parameters."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    golden = torch.nn.functional.rrelu(torch_input.float(), lower=0.125, upper=1.0 / 3.0, training=False).to(
        torch.bfloat16
    )

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input)
    tt_result = ttnn.to_torch(tt_output)

    assert_with_ulp(golden, tt_result, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
    ],
)
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),
        (0.0, 0.5),
    ],
)
def test_rrelu_eval_positive_inputs(input_shape, lower, upper, device):
    """Test rrelu eval mode with all-positive inputs (output should equal input)."""
    torch.manual_seed(0)
    torch_input = torch.abs(torch.randn(input_shape, dtype=torch.bfloat16)) + 0.01

    golden = torch_input.clone()

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    tt_result = ttnn.to_torch(tt_output)

    assert_with_ulp(golden, tt_result, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
    ],
)
def test_rrelu_eval_negative_inputs(input_shape, device):
    """Test rrelu eval mode with all-negative inputs (output = slope * input)."""
    torch.manual_seed(0)
    lower, upper = 0.125, 1.0 / 3.0
    torch_input = -torch.abs(torch.randn(input_shape, dtype=torch.bfloat16)) - 0.01

    golden = torch.nn.functional.rrelu(torch_input.float(), lower=lower, upper=upper, training=False).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    tt_result = ttnn.to_torch(tt_output)

    assert_with_ulp(golden, tt_result, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),
    ],
)
def test_rrelu_eval_l1_memory(input_shape, lower, upper, device):
    """Test rrelu eval mode with L1 memory config."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    golden = torch.nn.functional.rrelu(torch_input.float(), lower=lower, upper=upper, training=False).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_result = ttnn.to_torch(tt_output)

    assert_with_ulp(golden, tt_result, ulp_threshold=2)
