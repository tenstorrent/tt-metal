# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
  math_op, dim, scaler_mode, output_mem_config, output_dtype, compute_kernel_config,
  sub_core_grids, negate, program_factory.index(), input dtype,
  input memory_config, input padded_shape.

It deliberately EXCLUDES the two scalar floats (scaler / post_mul_scaler): they reach the
kernels as common runtime args, so distinct scalar values share one program (#54180), and
override_runtime_arguments() re-applies them on a cache hit.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_reduce_op(device, op, shape, dim, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, scalar=1.0):
    """Run a reduce op on device and return (torch_result, ttnn_result). ttnn(x, scalar=s) == torch(s * x)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_a = torch.rand(shape, dtype=torch_dtype) + 0.1

    ttnn_ops = {ttnn.sum: torch.sum, ttnn.max: torch.amax, ttnn.min: torch.amin, ttnn.mean: torch.mean}
    torch_result = ttnn_ops[op](scalar * torch_a, dim=dim, keepdim=True)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    with device.cache_entries_counter.measure():
        tt_result = op(tt_a, dim=dim, keepdim=True, memory_config=memory_config, scalar=scalar)
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
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(tt_out1, tt_out2)


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


def test_reduce_cache_miss_different_math_ops(device, isolate_program_cache):
    """Different reduce math ops (sum vs max) -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.max, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_dims(device, isolate_program_cache):
    """Different reduce dims (W vs H) -> different program factories -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    # dim=-1 (W): ReduceMultiCoreWProgramFactory
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    # dim=-2 (H): ReduceMultiCoreHProgramFactory
    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-2, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.float32)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.999,
        rtol=0.007,
        atol=0.25,
        frobenius_threshold=0.001,
        check_ulp=True,
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.999,
        rtol=0.004,
        atol=0.152,
        frobenius_threshold=0.003,
    )
    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_memory_configs(device, isolate_program_cache):
    """Different memory configs -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(
        device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    torch_ref2, tt_out2 = run_reduce_op(
        device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )

    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_shapes(device, isolate_program_cache):
    """Different padded shapes -> different cache entries.
    padded_shape is included in compute_program_hash() because Ht, Wt are compile-time args."""
    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, [1, 1, 32, 64], dim=-1, dtype=ttnn.bfloat16)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, [1, 1, 64, 64], dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )
    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_sub_core_grids(device, isolate_program_cache):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is in compute_program_hash() and affects work distribution (compile-time)."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]
    torch_a = torch.rand(shape, dtype=torch.bfloat16) + 0.1

    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out1 = ttnn.sum(tt_a, dim=-1, keepdim=True, sub_core_grids=grid_a)
        tt_out2 = ttnn.sum(tt_a, dim=-1, keepdim=True, sub_core_grids=grid_b)

    torch_ref = torch.sum(torch_a, dim=-1, keepdim=True)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref,
        ttnn.to_torch(tt_out1),
        pcc_threshold=0.999,
        rtol=0.007,
        atol=0.25,
        frobenius_threshold=0.001,
    )
    # test for equivalence
    assert_numeric_metrics(
        torch_ref,
        ttnn.to_torch(tt_out2),
        pcc_threshold=0.999,
        rtol=0.007,
        atol=0.25,
        frobenius_threshold=0.001,
    )

    assert device.cache_entries_counter.total == 2


# =============================================================================
# Scalar values are runtime args (#54180): one program must serve every value
# =============================================================================


@pytest.mark.parametrize("op", [ttnn.sum, ttnn.max, ttnn.min, ttnn.mean])
@pytest.mark.parametrize("dim", [-1, -2])
def test_reduce_cache_reuse_across_scalars(device, isolate_program_cache, op, dim):
    """Different scalar values -> 1 cache entry, and each result is correct."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    for scalar in [1.0, 0.5, 2.0]:
        torch_ref, tt_out = run_reduce_op(device, op, shape, dim=dim, scalar=scalar)
        # test for equivalance
        assert_numeric_metrics(
            torch_ref,
            tt_out,
            pcc_threshold=0.9999,
            rtol=1e-06,
            atol=1e-06,
            frobenius_threshold=1e-09,
        )

    assert device.cache_entries_counter.total == 1


def test_reduce_cache_reuse_across_scalar_signs_hw(device, isolate_program_cache):
    """Mixed-sign scalars on the HW factory -> 1 cache entry.

    The host used to hand REDUCE_SCALAR sqrt(scaler), which is NaN for a negative value, so
    `dim == HW && scaler < 0` was forced onto the two-step W-then-H path. The scalar is applied
    after the reduction now, so both signs share the one HW program: this cost 3 entries before
    (1 for the positive HW program, 2 for the forked W-then-H) and costs 1 now.
    """
    torch.manual_seed(0)
    shape = [1, 1, 32, 32]

    for scalar in [0.5, -0.5]:
        torch_ref, tt_out = run_reduce_op(device, ttnn.sum, shape, dim=[-2, -1], scalar=scalar)
        # test for equivalance
        assert_numeric_metrics(
            torch_ref,
            tt_out,
            pcc_threshold=0.9999,
            rtol=0.007,
            atol=0.25,
            frobenius_threshold=0.008,
        )

    assert device.cache_entries_counter.total == 1
