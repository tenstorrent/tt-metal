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
kernels as common runtime args, so distinct scalar values share one program (#54180).
scaler_mode stands in for the structural half — which slot is live, hence which defines the
kernels are built with — and is derived from op semantics alone, never from the value.

Each program factory re-applies all per-dispatch state on a cache hit via
override_runtime_arguments(), so shape/work distribution still requires separate cache
entries (padded_shape is in hash) but scalar values do not.
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


def run_reduce_op(device, op, shape, dim, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a reduce op on device and return (torch_result, ttnn_result)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_a = torch.rand(shape, dtype=torch_dtype) + 0.1

    ttnn_ops = {ttnn.sum: torch.sum, ttnn.max: torch.amax, ttnn.min: torch.amin}
    torch_result = ttnn_ops[op](torch_a, dim=dim, keepdim=True)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    with device.cache_entries_counter.measure():
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

# Positive scalars only: for max/min a negative scalar flips the op (see
# test_reduce_negative_scalar_flips_min_max), which is a different reduction, not a cache miss.
# 1.0 is the identity the kernels skip at runtime; 0.03125 is a representative mean-style 1/N.
SCALARS = [1.0, 0.5, 2.0, 0.03125]

TORCH_REDUCE = {"sum": torch.sum, "max": torch.amax, "min": torch.amin}
TTNN_REDUCE = {"sum": ttnn.sum, "max": ttnn.max, "min": ttnn.min}


def _reduce_and_check(device, op, dim, scalar, torch_input, ttnn_input):
    """Run one scaled reduce and assert it matches torch. ttnn_op(x, scalar=s) == torch_op(s * x)."""
    ttnn_result = ttnn.to_torch(TTNN_REDUCE[op](ttnn_input, dim=dim, scalar=scalar))
    torch_result = TORCH_REDUCE[op](scalar * torch_input, dim=dim)
    # Tolerances suit bf16/fp32 accumulation over signed inputs, where cancellation amplifies
    # relative error. They stay orders of magnitude tighter than a stale scalar would be, since
    # that is wrong by the ratio between two entries in SCALARS.
    assert_numeric_metrics(
        torch_result,
        ttnn_result,
        pcc_threshold=0.999,
        rtol=0.02,
        atol=0.3,
        frobenius_threshold=0.02,
    )


def _assert_scalars_share_one_program(device, op, dim, scalars, shape=(1, 1, 64, 64), dtype=torch.bfloat16):
    """Assert that N distinct scalars cost no more cache entries than a single scalar does.

    Asserted relative to a measured baseline rather than a hardcoded count: some dims lower to
    more than one op, and the invariant under test is that the entry count does not GROW with the
    number of distinct scalar values.
    """
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=dtype)
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # Baseline: what one scalar costs, whatever this dim happens to lower to.
    with device.cache_entries_counter.measure():
        _reduce_and_check(device, op, dim, scalars[0], torch_input, ttnn_input)
    baseline = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        for scalar in scalars[1:]:
            _reduce_and_check(device, op, dim, scalar, torch_input, ttnn_input)

    # The success criterion from #54180: same program, correct result per value.
    assert device.cache_entries_counter.total == baseline, (
        f"{len(scalars)} scalar values cost {device.cache_entries_counter.total} cache entries; "
        f"one scalar costs {baseline}. A scalar value is still reaching the program hash."
    )


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("dim", [-1, -2])
def test_reduce_cache_reuse_across_scalars(device, isolate_program_cache, op, dim):
    """Distinct scalar values share one cache entry on the W (dim=-1) and H (dim=-2) factories.

    Covers both scaler modes: sum takes ScalerTile (the scaler CB), while max/min take PostMul
    (GMPOOL keeps only the scaler's exponent for them). bf16 min additionally lowers to -MAX(-x),
    exercising the fused-negate compute kernels.
    """
    _assert_scalars_share_one_program(device, op, dim, SCALARS)


@pytest.mark.parametrize(
    "op, scalars",
    [
        # sum with scalar=1.0 additionally qualifies for the fast_reduce_nc fused path, which
        # cannot apply a scalar at all (see call_fast_nc in generic_reductions.cpp). Selecting a
        # different, faster op for unity is deliberate and unrelated to #54180, so 1.0 is excluded
        # here; the identity case is covered on the H/W dims and by the Int32 test below.
        ("sum", [0.5, 2.0, 0.03125, 0.25]),
        ("max", SCALARS),
    ],
)
def test_reduce_cache_reuse_across_scalars_hw(device, isolate_program_cache, op, scalars):
    """Distinct scalar values share one cache entry on the HW path.

    Positive scalars only: a negative scalar still forks HW onto the two-step W-then-H path because
    the single-core HW kernel pre-compensates with sqrt(scaler). Removing that fork is Phase 2 of
    #54180; until then a mixed-sign HW set legitimately needs two entries.
    """
    _assert_scalars_share_one_program(device, op, dim=None, scalars=scalars)


def test_reduce_cache_reuse_across_scalars_float32(device, isolate_program_cache):
    """Distinct scalar values share one cache entry on the fp32 path."""
    _assert_scalars_share_one_program(device, "sum", dim=-1, scalars=SCALARS, dtype=torch.float32)


def test_reduce_sum_cache_reuse_across_scalar_signs(device, isolate_program_cache):
    """For sum, scalar sign does not change the reduction, so signs share one cache entry too."""
    _assert_scalars_share_one_program(device, "sum", dim=-1, scalars=[1.0, 0.5, -3.0, -0.25])


@pytest.mark.parametrize("op", ["max", "min"])
def test_reduce_negative_scalar_flips_min_max(device, isolate_program_cache, op):
    """A negative scalar turns max into min (and back), so it needs its own cache entry.

    The scalar is applied after the reduction, and max(s*x) == s*min(x) for s < 0, so the
    dispatcher flips the op. That is a different reduction, not a stale-hash cache miss: exactly
    one extra entry appears, and both results are correct.
    """
    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    with device.cache_entries_counter.measure():
        _reduce_and_check(device, op, -1, 2.0, torch_input, ttnn_input)
    after_positive = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        _reduce_and_check(device, op, -1, -3.0, torch_input, ttnn_input)
    after_first_negative = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        _reduce_and_check(device, op, -1, -0.25, torch_input, ttnn_input)
    after_second_negative = device.cache_entries_counter.total

    assert after_first_negative > after_positive, "the flipped op should need its own program"
    assert (
        after_second_negative == after_first_negative
    ), "two negative scalars differ only in value, so they must share the flipped op's program"


def test_reduce_int32_identity_scalar_is_not_lossy(device, isolate_program_cache):
    """An Int32 reduce with scalar=1.0 must stay bit-exact above 2^24.

    Int32 post-multiply brackets mul_unary_tile with Int32<->Float32 typecasts, and fp32 has only
    24 mantissa bits. Since one program now serves every scalar, the post-multiply is compiled in
    even when the caller passes no scalar, so the kernels must skip a 1.0f scalar at runtime rather
    than execute a lossy multiply-by-one. This asserts that skip is in place.
    """
    # 31 * 2^20 + (2^20 + 1) == 2^25 + 1, which is NOT representable in fp32 (ulp at 2^25 is 4),
    # so a round trip through float32 would return 2^25 instead.
    row = torch.full((32,), 1048576, dtype=torch.int32)
    row[0] = 1048577
    torch_input = row.reshape(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    expected = 2**25 + 1

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    ttnn_result = ttnn.to_torch(ttnn.sum(ttnn_input, dim=-1, keepdim=True, scalar=1.0))

    assert torch.all(ttnn_result == expected), (
        f"Int32 sum with scalar=1.0 returned {ttnn_result.flatten()[0].item()}, expected {expected}. "
        "A multiply-by-one survived the runtime identity check and truncated through fp32."
    )


def _height_sharded_config(shard_h, shard_w, num_cores):
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shard_w),
        core_grid=ttnn.CoreGrid(x=num_cores, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def test_reduce_cache_reuse_across_scalars_height_sharded(device, isolate_program_cache):
    """Distinct scalar values share one cache entry on the height-sharded W path.

    This is the branch whose input and output shards are aliased as circular buffers instead of
    being passed as runtime-arg addresses, so it is the branch whose cache-hit path re-points CB
    addresses. Covered here because a stale CB address would survive every interleaved test.
    """
    shape = (1, 1, 256, 128)
    num_cores = 8
    input_config = _height_sharded_config(shape[2] // num_cores, shape[3], num_cores)
    output_config = _height_sharded_config(shape[2] // num_cores, 32, num_cores)

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_config)

    def run(scalar):
        result = ttnn.to_torch(ttnn.sum(ttnn_input, dim=-1, keepdim=True, scalar=scalar, memory_config=output_config))
        expected = torch.sum(scalar * torch_input, dim=-1, keepdim=True)
        assert_numeric_metrics(expected, result, pcc_threshold=0.999, rtol=0.02, atol=0.3, frobenius_threshold=0.02)

    with device.cache_entries_counter.measure():
        run(SCALARS[0])
    baseline = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        for scalar in SCALARS[1:]:
            run(scalar)

    assert device.cache_entries_counter.total == baseline, (
        f"{len(SCALARS)} scalar values cost {device.cache_entries_counter.total} cache entries; "
        f"one scalar costs {baseline}."
    )
