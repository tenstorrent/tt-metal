# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for output_mem_config propagation in ttnn.binary_backward ops (#65472)."""

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
)
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
    grad_data, grad_tensor = data_gen_with_range(shape, -5, 5, device, seed=42)
    in_data, input_tensor = data_gen_with_range(shape, -100, 100, device, required_grad=True, seed=43)
    other_data, other_tensor = data_gen_with_range(shape, -10, 10, device, required_grad=True, seed=44)

    grad_tt = ttnn.to_memory_config(grad_tensor, in_mem_config)
    input_tt = ttnn.to_memory_config(input_tensor, in_mem_config)
    other_tt = ttnn.to_memory_config(other_tensor, in_mem_config)

    out = ttnn.squared_difference_bw(grad_tt, input_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.squared_difference_bw)
    golden = golden_fn(grad_data, in_data, other_data)
    assert compare_pcc(out, golden, pcc=0.999)


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
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16)
    in_b_pt = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_b_tt = ttnn.from_torch(in_b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.bias_gelu_bw(grad_tt, in_a_tt, in_b_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), in_b_pt.float())
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
    in_pt = torch.randn(shape, dtype=torch.bfloat16)
    bias = 2.5

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_tt = ttnn.from_torch(in_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.bias_gelu_bw(grad_tt, in_tt, bias, memory_config=out_mem_config)

    assert len(out) == 1
    assert out[0].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    golden = golden_fn(grad_pt.float(), in_pt.float(), bias)
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
    in_a_pt = torch.rand(shape, dtype=torch.bfloat16) * 10 + 1
    other_pt = torch.rand(shape, dtype=torch.bfloat16) * 5 + 1
    other_pt[other_pt == 0] = 1.0

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.remainder_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    # out[0] is grad_tensor returned directly (preserves in_mem_config)
    assert out[0].memory_config() == in_mem_config
    # out[1] is computed gradient with intermediate neg(grad_tensor) honoring out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.remainder_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), other_pt.float())
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
    in_a_pt = torch.rand(shape, dtype=torch.bfloat16) * 10 + 1
    other_pt = torch.rand(shape, dtype=torch.bfloat16) * 5 + 1
    other_pt[other_pt == 0] = 1.0

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.fmod_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    # out[0] is grad_tensor returned directly (preserves in_mem_config)
    assert out[0].memory_config() == in_mem_config
    # out[1] is computed gradient with intermediate neg(grad_tensor) honoring out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.fmod_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), other_pt.float())
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
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16)
    other_pt = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.atan2_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.atan2_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), other_pt.float())
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
    in_a_pt = torch.randn(shape, dtype=torch.bfloat16)
    other_pt = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(grad_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    in_a_tt = ttnn.from_torch(in_a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)
    other_tt = ttnn.from_torch(other_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=in_mem_config)

    out = ttnn.logaddexp_bw(grad_tt, in_a_tt, other_tt, memory_config=out_mem_config)

    assert len(out) == 2
    assert out[0].memory_config() == out_mem_config
    assert out[1].memory_config() == out_mem_config

    golden_fn = ttnn.get_golden_function(ttnn.logaddexp_bw)
    golden = golden_fn(grad_pt.float(), in_a_pt.float(), other_pt.float())
    assert_with_pcc(golden[0], ttnn.to_torch(out[0]), 0.999)
    assert_with_pcc(golden[1], ttnn.to_torch(out[1]), 0.999)
