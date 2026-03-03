# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for normalization/softmax program cache behavior.

Tests target potential caching issues.
SoftmaxDeviceOperation uses 7 ProgramFactory variants selected by select_program_factory():
  - SoftmaxProgramFactoryGeneralWSmall / WLarge: Softmax, last dim, 5D+ tensor
  - SoftmaxProgramFactoryGeneralHSmall / HLarge: Softmax, second-to-last dim, 5D+ tensor
  - SoftmaxProgramFactoryGeneralCLarge: Softmax, other dim, 5D+ tensor
  - SoftmaxProgramFactoryAttentionOptimized: Softmax/ScaleMaskSoftmax/inplace variants
  - SoftmaxShardedProgramFactoryAttentionOptimized: sharded config

compute_program_hash() includes (after fix):
  program_factory.index(), softmax_type, dim, scale, inplace, output_mem_config,
  program_config, is_causal_mask, compute_kernel_config, is_scale_causal_mask_hw_dims,
  numeric_stable, input logical_shape, input dtype, input memory_config, input layout,
  mask dtype (if present), mask memory_config (if present), mask padded_shape (if present).

Mask tensor properties were added to the hash because:
  - mask dtype → affects CB data format (compile-time configuration)
  - mask memory_config → affects TensorAccessorArgs (compile-time args)
  - mask padded_shape → num_tiles_causal_mask for causal mask (compile-time arg)
"""

import os
import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_wormhole_b0


def is_simulator():
    return os.environ.get("TT_METAL_SIMULATOR") != None


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_softmax_5d(device, shape, dim, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run softmax on a 5D tensor (uses general factories, not attention optimized)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_a = torch.rand(shape, dtype=torch_dtype)
    torch_result = F.softmax(torch_a, dim=dim)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_result = ttnn.softmax(tt_a, dim=dim, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


def run_softmax_4d(device, shape, dim, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run softmax on a 4D tensor (uses attention optimized factory for last dim)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_a = torch.rand(shape, dtype=torch_dtype)
    torch_result = F.softmax(torch_a, dim=dim)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_result = ttnn.softmax(tt_a, dim=dim, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


# =============================================================================
# Cache reuse tests
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_reuse_same_config_5d(device, isolate_program_cache):
    """Same op, same 5D shape, same dtype run twice -> 1 cache entry, different outputs."""
    shape = [1, 1, 1, 32, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_softmax_5d(device, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_softmax_5d(device, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(tt_out1, tt_out2)


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_reuse_same_config_4d(device, isolate_program_cache):
    """Same op, same 4D shape with last dim -> 1 cache entry (attention optimized factory)."""
    shape = [1, 1, 32, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_softmax_4d(device, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_softmax_4d(device, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(tt_out1, tt_out2)


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_miss_different_dims_5d(device, isolate_program_cache):
    """Different dims on 5D tensor -> different factories -> different cache entries.
    dim=-1 -> WSmall/WLarge, dim=-2 -> HSmall/HLarge, dim=-3 -> CLarge."""
    shape = [1, 1, 2, 32, 64]

    torch_ref1, tt_out1 = run_softmax_5d(device, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_softmax_5d(device, shape, dim=-2, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_miss_different_factories(device, isolate_program_cache):
    """5D W-softmax vs 4D last-dim softmax -> different factories -> different cache entries."""
    shape_5d = [1, 1, 1, 32, 64]
    shape_4d = [1, 1, 32, 64]

    # 5D: general W factory
    torch_ref1, tt_out1 = run_softmax_5d(device, shape_5d, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    # 4D last dim: attention optimized factory
    torch_ref2, tt_out2 = run_softmax_4d(device, shape_4d, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_softmax_4d(device, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_softmax_4d(device, shape, dim=-1, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_miss_different_memory_configs(device, isolate_program_cache):
    """Different memory configs -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_softmax_4d(
        device, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_softmax_4d(
        device, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_softmax_cache_miss_different_shapes(device, isolate_program_cache):
    """Different logical shapes -> different cache entries.
    logical_shape is in compute_program_hash() determining Wt, Ht, work distribution."""
    torch_ref1, tt_out1 = run_softmax_4d(device, [1, 1, 32, 64], dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_softmax_4d(device, [1, 1, 64, 64], dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


# =============================================================================
# Mask tensor property tests (verifying the hash fix)
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_scale_mask_softmax_cache_miss_different_mask_dtypes(device, isolate_program_cache):
    """scale_mask_softmax with masks of different dtypes -> different cache entries.

    mask dtype affects mask_cb_data_format (CB format), which is a compile-time configuration
    embedded in the program. Before the fix, both calls would hit the same cache entry,
    potentially using the wrong CB format for the second mask dtype.
    """
    batch = 1
    seq = 32
    inner = 64
    input_shape = [batch, 1, seq, inner]
    mask_shape = [batch, 1, 1, inner]

    torch_a = torch.rand(input_shape, dtype=torch.bfloat16)

    # First call: mask in bfloat16
    torch_mask_bf16 = torch.zeros(mask_shape, dtype=torch.bfloat16)
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask_bf16 = ttnn.from_torch(torch_mask_bf16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.scale_mask_softmax(tt_a, scale=None, mask=tt_mask_bf16)
    torch_ref1 = ttnn.to_torch(tt_out1)
    assert_with_pcc(F.softmax(torch_a, dim=-1), torch_ref1, 0.999)

    # Second call: mask in float32 (different dtype)
    torch_mask_fp32 = torch.zeros(mask_shape, dtype=torch.float32)
    tt_a2 = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask_fp32 = ttnn.from_torch(torch_mask_fp32, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    tt_out2 = ttnn.scale_mask_softmax(tt_a2, scale=None, mask=tt_mask_fp32)
    torch_ref2 = ttnn.to_torch(tt_out2)
    assert_with_pcc(F.softmax(torch_a, dim=-1), torch_ref2, 0.999)

    # Different mask dtypes must produce different cache entries
    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_scale_mask_softmax_cache_miss_different_mask_memory_configs(device, isolate_program_cache):
    """scale_mask_softmax with masks in different memory locations -> different cache entries.

    mask memory_config determines the TensorAccessorArgs (IsDram compile-time flag).
    DRAM accessor (IsDram=1) vs L1 accessor (IsDram=0) require different compiled kernels.
    Before the fix, both calls would hit the same cache entry using the wrong accessor
    for the second call.
    """
    batch = 1
    seq = 32
    inner = 64
    input_shape = [batch, 1, seq, inner]
    mask_shape = [batch, 1, 1, inner]

    torch_a = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_mask = torch.zeros(mask_shape, dtype=torch.bfloat16)

    # First call: mask in DRAM (default)
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask_dram = ttnn.from_torch(
        torch_mask, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out1 = ttnn.scale_mask_softmax(tt_a, scale=None, mask=tt_mask_dram)
    assert_with_pcc(F.softmax(torch_a, dim=-1), ttnn.to_torch(tt_out1), 0.999)

    # Second call: mask in L1 (different memory config)
    tt_a2 = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask_l1 = ttnn.from_torch(
        torch_mask, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_out2 = ttnn.scale_mask_softmax(tt_a2, scale=None, mask=tt_mask_l1)
    assert_with_pcc(F.softmax(torch_a, dim=-1), ttnn.to_torch(tt_out2), 0.999)

    # Different mask memory configs must produce different cache entries
    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_scale_mask_softmax_cache_reuse_same_mask_config(device, isolate_program_cache):
    """scale_mask_softmax with same mask config twice -> 1 cache entry, different outputs."""
    batch = 1
    seq = 32
    inner = 64
    input_shape = [batch, 1, seq, inner]
    mask_shape = [batch, 1, 1, inner]

    torch.manual_seed(0)
    torch_a1 = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_mask1 = torch.zeros(mask_shape, dtype=torch.bfloat16)
    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask1 = ttnn.from_torch(torch_mask1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.scale_mask_softmax(tt_a1, scale=None, mask=tt_mask1)
    torch_ref1 = ttnn.to_torch(tt_out1)

    torch.manual_seed(42)
    torch_a2 = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_mask2 = torch.zeros(mask_shape, dtype=torch.bfloat16)
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask2 = ttnn.from_torch(torch_mask2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = ttnn.scale_mask_softmax(tt_a2, scale=None, mask=tt_mask2)
    torch_ref2 = ttnn.to_torch(tt_out2)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(torch_ref1, torch_ref2)
