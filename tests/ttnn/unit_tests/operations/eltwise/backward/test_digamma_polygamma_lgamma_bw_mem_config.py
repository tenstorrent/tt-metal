# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for digamma_bw, polygamma_bw, and lgamma_bw memory config forwarding.

All three functions resolve output_memory_config from the raw optional via
.value_or(input.memory_config()), but then incorrectly passed the raw
output_mem_config optional to inner op calls. This meant the explicit memory
config was silently ignored and inner ops used the default resolution path.
"""

import pytest
import torch
import ttnn

torch.manual_seed(0)


def test_digamma_bw_memory_config_preserved(device):
    """Verify digamma_bw output lands on the requested memory config."""
    shape = [1, 1, 32, 32]
    input_torch = torch.rand(shape, dtype=torch.bfloat16) + 1.0
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dram_config = ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.digamma_bw(grad_tt, input_tt, memory_config=dram_config)
    assert result[0].memory_config() == dram_config, (
        f"digamma_bw: expected DRAM, got {result[0].memory_config()}"
    )


def test_digamma_bw_correctness(device):
    """Test digamma_bw produces correct gradients vs PyTorch reference.

    digamma_bw(grad, x) = grad * polygamma(1, x) for x > 0
    Special cases at x=0 handled separately.
    """
    shape = [1, 1, 32, 32]
    input_torch = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.digamma_bw(grad_tt, input_tt)
    result_torch = ttnn.to_torch(result[0])

    expected = grad_torch * torch.special.polygamma(1, input_torch)

    assert torch.allclose(result_torch.float(), expected.float(), atol=2e-2), (
        f"digamma_bw correctness: max diff {(result_torch - expected).abs().max()}"
    )


def test_lgamma_bw_memory_config_preserved(device):
    """Verify lgamma_bw output lands on the requested memory config."""
    shape = [1, 1, 32, 32]
    input_torch = torch.rand(shape, dtype=torch.bfloat16) + 1.0
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dram_config = ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.lgamma_bw(grad_tt, input_tt, memory_config=dram_config)
    assert result[0].memory_config() == dram_config, (
        f"lgamma_bw: expected DRAM, got {result[0].memory_config()}"
    )


def test_lgamma_bw_correctness(device):
    """Test lgamma_bw produces correct gradients vs PyTorch reference.

    lgamma_bw(grad, x) = grad * digamma(x)
    """
    shape = [1, 1, 32, 32]
    input_torch = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.lgamma_bw(grad_tt, input_tt)
    result_torch = ttnn.to_torch(result[0])

    expected = grad_torch * torch.special.digamma(input_torch)

    assert torch.allclose(result_torch.float(), expected.float(), atol=2e-2), (
        f"lgamma_bw correctness: max diff {(result_torch - expected).abs().max()}"
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_polygamma_bw_memory_config_preserved(device, n):
    """Verify polygamma_bw output lands on the requested memory config."""
    shape = [1, 1, 32, 32]
    input_torch = torch.rand(shape, dtype=torch.bfloat16) + 1.0
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dram_config = ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.polygamma_bw(grad_tt, input_tt, n, memory_config=dram_config)
    assert result[0].memory_config() == dram_config, (
        f"polygamma_bw(n={n}): expected DRAM, got {result[0].memory_config()}"
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_polygamma_bw_correctness(device, n):
    """Test polygamma_bw produces correct gradients vs PyTorch reference.

    polygamma_bw(grad, x, n) = grad * polygamma(n+1, x)
    """
    shape = [1, 1, 32, 32]
    input_torch = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    grad_torch = torch.ones(shape, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.polygamma_bw(grad_tt, input_tt, n)
    result_torch = ttnn.to_torch(result[0])

    expected = grad_torch * torch.special.polygamma(n + 1, input_torch)

    assert torch.allclose(result_torch.float(), expected.float(), atol=2e-2), (
        f"polygamma_bw(n={n}) correctness: max diff {(result_torch - expected).abs().max()}"
    )
