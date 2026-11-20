# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for output_mem_config parameter acceptance, returned buffer memory configs,
and PyTorch autograd golden PCC verification for ttnn.binary_backward ops (#65472).

Note: These unit tests verify end-to-end memory_config handling on returned tensors and
compare numeric outputs against PyTorch autograd references (PCC >= 0.999). Interleaved
L1/DRAM memory configuration checks verify returned tensor placement and op execution
across cross-memory configurations, but cannot directly observe internal intermediate buffer
placement (e.g. intermediate neg(grad_tensor) in remainder_bw/fmod_bw or intermediate
subtract/add buffers) without device-level calltrace/memory inspection or sharded layout
specifications.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_squared_difference_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    in_b_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_b_tt = ttnn.from_torch(in_b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.squared_difference_bw(grad_tt, in_a_tt, in_b_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.squared_difference_bw)
    golden = golden_fn(grad_pt, in_a_pt, in_b_pt)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_bias_gelu_tensor_bias_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    in_b_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_b_tt = ttnn.from_torch(in_b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.bias_gelu_bw(grad_tt, in_a_tt, in_b_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    golden = golden_fn(grad_pt, in_a_pt, in_b_pt)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_bias_gelu_scalar_bias_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    bias = 2.5

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_tt = ttnn.from_torch(in_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.bias_gelu_bw(grad_tt, in_tt, bias, memory_config=out_mem_config)

    assert len(out) == 1
    assert out[0].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    golden = golden_fn(grad_pt, in_pt, bias)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_remainder_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = (torch.rand(shape, dtype=torch.bfloat16) * 10 + 1).requires_grad_(True)
    other_pt = (torch.rand(shape, dtype=torch.bfloat16) * 5 + 1).requires_grad_(True)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.remainder_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    # out[0] is grad_tensor returned directly (preserves in_mem_config)
    assert out[0].memory_config() == in_mem_config
    # out[1] has output_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.remainder_bw)
    golden = golden_fn(grad_pt, in_a_pt, other_pt)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_fmod_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = (torch.rand(shape, dtype=torch.bfloat16) * 10 + 1).requires_grad_(True)
    other_pt = (torch.rand(shape, dtype=torch.bfloat16) * 5 + 1).requires_grad_(True)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.fmod_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    # out[0] is grad_tensor returned directly (preserves in_mem_config)
    assert out[0].memory_config() == in_mem_config
    # out[1] has output_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.fmod_bw)
    golden = golden_fn(grad_pt, in_a_pt, other_pt)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_atan2_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    other_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.atan2_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.atan2_bw)
    golden = golden_fn(grad_pt, in_a_pt, other_pt)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "in_mem_config, out_mem_config",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_bw_logaddexp_mem_config(device, in_mem_config, out_mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    other_pt = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.logaddexp_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.logaddexp_bw)
    golden = golden_fn(grad_pt, in_a_pt, other_pt)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)
