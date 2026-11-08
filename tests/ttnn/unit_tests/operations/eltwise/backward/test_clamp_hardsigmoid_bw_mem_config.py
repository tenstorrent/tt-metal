# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for clamp_bw and hardsigmoid_bw memory config forwarding bugs.

clamp_bw bug: The !max and !min early-return branches used the raw
output_mem_config optional instead of the resolved output_memory_config,
causing incorrect memory placement when output_mem_config is std::nullopt.

hardsigmoid_bw bug: The intermediate ttnn::multiply(grad, 1.0f/6.0f) did not
forward output_mem_config, causing the intermediate to inherit grad's memory
config instead of the requested output config.
"""

import pytest
import torch
import ttnn

torch.manual_seed(0)


@pytest.mark.parametrize("min_val", [None, -0.5, 0.0])
@pytest.mark.parametrize("max_val", [0.5, 1.0, None])
def test_clamp_bw_float_minmax_branches(device, min_val, max_val):
    """Test clamp_bw with float min/max — exercises all three branches.

    Before the fix, the !max branch (max_val=None) and !min branch (min_val=None)
    used the wrong memory config variable.
    """
    if min_val is None and max_val is None:
        pytest.skip("clamp requires at least one of min/max")

    shape = [1, 1, 32, 32]
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.clamp_bw(grad_tt, input_tt, min_val, max_val)
    result_torch = ttnn.to_torch(result[0])

    # Compute expected: grad passes through where min <= input <= max
    expected = grad_torch.clone()
    if min_val is not None:
        expected[input_torch < min_val] = 0.0
    if max_val is not None:
        expected[input_torch > max_val] = 0.0

    assert torch.allclose(result_torch.float(), expected.float(), atol=1e-2), (
        f"clamp_bw(min={min_val}, max={max_val}) mismatch"
    )


@pytest.mark.parametrize("min_val", [None, -0.5])
@pytest.mark.parametrize("max_val", [0.5, None])
def test_clamp_bw_tensor_minmax_branches(device, min_val, max_val):
    """Test clamp_bw with Tensor min/max — exercises all three branches of overload 2.

    Before the fix, the !max and !min branches used output_mem_config instead of
    the resolved output_memory_config.
    """
    if min_val is None and max_val is None:
        pytest.skip("clamp requires at least one of min/max")

    shape = [1, 1, 32, 32]
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    min_tt = None
    max_tt = None
    min_ref = None
    max_ref = None

    if min_val is not None:
        min_torch = torch.full(shape, min_val, dtype=torch.bfloat16)
        min_tt = ttnn.from_torch(min_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        min_ref = min_torch
    if max_val is not None:
        max_torch = torch.full(shape, max_val, dtype=torch.bfloat16)
        max_tt = ttnn.from_torch(max_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        max_ref = max_torch

    result = ttnn.clamp_bw(grad_tt, input_tt, min_tt, max_tt)
    result_torch = ttnn.to_torch(result[0])

    expected = grad_torch.clone()
    if min_ref is not None:
        expected[input_torch < min_ref] = 0.0
    if max_ref is not None:
        expected[input_torch > max_ref] = 0.0

    assert torch.allclose(result_torch.float(), expected.float(), atol=1e-2), (
        f"clamp_bw tensor(min={min_val}, max={max_val}) mismatch"
    )


def test_clamp_bw_memory_config_preserved(device):
    """Verify clamp_bw output lands on the requested memory config.

    Before the fix, the !max/!min branches used the raw optional (std::nullopt),
    which resolved to input's config instead of DRAM when explicitly requested.
    """
    shape = [1, 1, 32, 32]
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dram_config = ttnn.DRAM_MEMORY_CONFIG

    # Test !max branch (max_val=None)
    result = ttnn.clamp_bw(grad_tt, input_tt, -0.5, None, output_mem_config=dram_config)
    assert result[0].memory_config() == dram_config, (
        f"clamp_bw !max branch: expected DRAM, got {result[0].memory_config()}"
    )

    # Test !min branch (min_val=None)
    result = ttnn.clamp_bw(grad_tt, input_tt, None, 0.5, output_mem_config=dram_config)
    assert result[0].memory_config() == dram_config, (
        f"clamp_bw !min branch: expected DRAM, got {result[0].memory_config()}"
    )


@pytest.mark.parametrize(
    "input_val",
    [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0],
)
def test_hardsigmoid_bw_correctness(device, input_val):
    """Test hardsigmoid_bw produces correct gradients.

    hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
    hardsigmoid_bw(grad, x) = 0 if |x| >= 3, else grad/6

    The fix ensures the intermediate multiply(grad, 1/6) forwards output_mem_config.
    """
    shape = [1, 1, 32, 32]
    input_torch = torch.full(shape, input_val, dtype=torch.bfloat16)
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.hardsigmoid_bw(grad_tt, input_tt)
    result_torch = ttnn.to_torch(result[0])

    # Expected: grad/6 if |x| < 3, else 0
    expected_val = 1.0 / 6.0 if abs(input_val) < 3.0 else 0.0
    expected = torch.full(shape, expected_val, dtype=torch.bfloat16)

    assert torch.allclose(result_torch.float(), expected.float(), atol=1e-2), (
        f"hardsigmoid_bw(x={input_val}): got {result_torch.flatten()[0].item()}, expected {expected_val}"
    )


def test_hardsigmoid_bw_memory_config_preserved(device):
    """Verify hardsigmoid_bw output lands on the requested memory config."""
    shape = [1, 1, 32, 32]
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dram_config = ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.hardsigmoid_bw(grad_tt, input_tt, output_mem_config=dram_config)
    assert result[0].memory_config() == dram_config, (
        f"hardsigmoid_bw: expected DRAM, got {result[0].memory_config()}"
    )
