# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for reduction/generic program cache behavior.

Tests target potential caching issues.
The ReduceDeviceOperation uses 3 ProgramFactory variants:
  - ReduceMultiCoreHProgramFactory (dim=H)
  - ReduceMultiCoreWProgramFactory (dim=W)
  - ReduceSingleCoreHwProgramFactory (dim=HW with single tile), or
    MULTI_CORE_HW which also maps to ReduceSingleCoreHwProgramFactory

compute_program_hash() includes:
  math_op, dim, scaler, output_mem_config, output_dtype, compute_kernel_config,
  sub_core_grids, negate, program_factory.index(), input dtype,
  input memory_config, input logical_shape, input padded_shape.

override_runtime_arguments() only updates buffer addresses — shape/work distribution
changes require separate cache entries (padded_shape is in hash).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_reduce_op(device, op, shape, dim, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a reduce op on device and return (torch_result, ttnn_result)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_a = torch.rand(shape, dtype=torch_dtype) + 0.1

    ttnn_ops = {ttnn.sum: torch.sum, ttnn.max: torch.amax, ttnn.min: torch.amin}
    torch_result = ttnn_ops[op](torch_a, dim=dim, keepdim=True)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_result = op(tt_a, dim=dim, keepdim=True, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


# =============================================================================
# Cache reuse tests (fields correctly excluded from hash)
# =============================================================================


def test_reduce_cache_reuse_same_config(device, isolate_program_cache):
    """Same op, same shape, same dtype run twice -> 1 cache entry, different outputs."""
    shape = [1, 1, 64, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(tt_out1, tt_out2)


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


def test_reduce_cache_miss_different_math_ops(device, isolate_program_cache):
    """Different reduce math ops (sum vs max) -> different cache entries."""
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.max, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


def test_reduce_cache_miss_different_dims(device, isolate_program_cache):
    """Different reduce dims (W vs H) -> different program factories -> different cache entries."""
    shape = [1, 1, 64, 64]

    # dim=-1 (W): ReduceMultiCoreWProgramFactory
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    # dim=-2 (H): ReduceMultiCoreHProgramFactory
    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-2, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


def test_reduce_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries."""
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


def test_reduce_cache_miss_different_memory_configs(device, isolate_program_cache):
    """Different memory configs -> different cache entries."""
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(
        device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_reduce_op(
        device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


def test_reduce_cache_miss_different_shapes(device, isolate_program_cache):
    """Different padded shapes -> different cache entries.
    padded_shape is included in compute_program_hash() because Ht, Wt are compile-time args."""
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, [1, 1, 32, 64], dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, [1, 1, 64, 64], dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


def test_reduce_cache_miss_same_padded_shape_different_logical_shape(device, isolate_program_cache):
    """Different logical shapes that share a padded shape still need separate cache entries.

    Native reduce padding bakes the logical tail sizes into reader compile-time args.
    Reusing a program across the same padded tile shape but different logical shapes
    would apply the wrong terminal-tile masking.
    """
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, [1, 1, 9, 9], dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, [1, 1, 37, 9], dim=-1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 2


def test_reduce_cache_miss_sub_core_grids(device, isolate_program_cache):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is in compute_program_hash() and affects work distribution (compile-time)."""
    shape = [1, 1, 64, 64]
    torch_a = torch.rand(shape, dtype=torch.bfloat16) + 0.1

    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.sum(tt_a, dim=-1, keepdim=True, sub_core_grids=grid_a)
    tt_out2 = ttnn.sum(tt_a, dim=-1, keepdim=True, sub_core_grids=grid_b)

    torch_ref = torch.sum(torch_a, dim=-1, keepdim=True)
    assert_with_pcc(torch_ref, ttnn.to_torch(tt_out1), 0.999)
    assert_with_pcc(torch_ref, ttnn.to_torch(tt_out2), 0.999)

    assert device.num_program_cache_entries() == 2
