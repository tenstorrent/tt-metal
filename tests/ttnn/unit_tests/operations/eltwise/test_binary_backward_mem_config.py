# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for output_mem_config propagation in ttnn.binary_backward ops (#65472)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_squared_difference_bw_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16)
    other_pt = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.squared_difference_bw(grad_tt, in_a_tt, other_tt, memory_config=mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == mem_config
    assert out[1].memory_config() == mem_config

    golden_fn = ttnn.get_golden_function(ttnn.squared_difference_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), other_pt.float())
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_bias_gelu_bw_tensor_bias_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_b_pt = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_b_tt = ttnn.from_torch(in_b_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.bias_gelu_bw(grad_tt, in_a_tt, in_b_tt, memory_config=mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == mem_config
    assert out[1].memory_config() == mem_config

    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), in_b_pt.float())
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_bias_gelu_bw_scalar_bias_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_pt = torch.randn(shape, dtype=torch.bfloat16)
    bias = 2.5

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_tt = ttnn.from_torch(in_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.bias_gelu_bw(grad_tt, in_tt, bias, memory_config=mem_config)

    assert len(out) == 1
    assert out[0].memory_config() == mem_config

    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    golden = golden_fn(grad_pt.float(), in_pt.float(), bias)
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_remainder_bw_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.rand(shape, dtype=torch.bfloat16) * 10 + 1
    other_pt = torch.rand(shape, dtype=torch.bfloat16) * 5 + 1

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.remainder_bw(grad_tt, in_a_tt, other_tt, memory_config=mem_config)

    assert len(out) == 2
    assert out[1].memory_config() == mem_config


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_fmod_bw_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.rand(shape, dtype=torch.bfloat16) * 10 + 1
    other_pt = torch.rand(shape, dtype=torch.bfloat16) * 5 + 1

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.fmod_bw(grad_tt, in_a_tt, other_tt, memory_config=mem_config)

    assert len(out) == 2
    assert out[1].memory_config() == mem_config


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_atan2_bw_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16) + 2.0
    other_pt = torch.randn(shape, dtype=torch.bfloat16) + 2.0

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.atan2_bw(grad_tt, in_a_tt, other_tt, memory_config=mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == mem_config
    assert out[1].memory_config() == mem_config


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
def test_logaddexp_bw_mem_config(device, mem_config):
    shape = [1, 1, 64, 64]
    torch.manual_seed(42)
    grad_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16)
    other_pt = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out = ttnn.logaddexp_bw(grad_tt, in_a_tt, other_tt, memory_config=mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == mem_config
    assert out[1].memory_config() == mem_config
