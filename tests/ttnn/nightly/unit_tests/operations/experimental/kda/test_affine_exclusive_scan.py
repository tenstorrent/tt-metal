# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for the experimental KDA affine exclusive scan."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    collect_accuracy_and_determinism_results,
    assert_equal,
)

from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    group_summary_nd_sharded,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import height_sharded_memory_config

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    batch_heads: int
    groups_per_head: int
    key_dim: int
    value_dim: int
    expected_duration_ns: int


_PRODUCTION_PERF_MARGIN = 0.05
# Reference recalibrated from 8,087 ns to the measured median of the no-alias
# implementation. Removing DFB aliasing costs 2 extra physical buffers and
# 12,288 bytes of worker L1, and that structural cost is accepted, so the
# reference tracks the accepted implementation rather than its aliased ancestor.
_UNIT_CASE = _ProductionCase(
    "bh2-g4-k32-v64",
    batch_heads=2,
    groups_per_head=4,
    key_dim=32,
    value_dim=64,
    expected_duration_ns=8250,
)

# Kimi-K3 production layouts. References are medians of the CI bh_p150b_civ2_viommu runner (four main nightlies,
# plus one PR run for sp1-tp8 and sp2-tp4); the earlier development-machine references sat at the top of the
# runner's spread, so sp2-tp4 fell below its band on an ordinary run (72,849 ns).
_PRODUCTION_CASES = (
    _ProductionCase("sp1-tp8", 12, 8, 128, 128, 99597),
    _ProductionCase("sp2-tp4", 24, 4, 128, 128, 74808),
    _ProductionCase("sp4-tp2", 48, 2, 128, 128, 65831),
)
# The grouped model's summaries: BF16 head and tail slots in L1, one ND shard per group-head on its worker's core.
# References are medians of five runs on the Blackhole LoudBox development machine bh-lb-42, whose DRAM-placement
# medians in the same runs (97,625 / 75,386 / 65,397 ns) lie within 2% of the CI references above. Replace them with
# CI runner medians once nightlies report them.
_ND_SHARDED_DURATION_NS = {"sp1-tp8": 90327, "sp2-tp4": 71042, "sp4-tp2": 59443}


def _affine_exclusive_scan_ops(
    summaries: torch.Tensor | ttnn.Tensor,
    initial_state: torch.Tensor | ttnn.Tensor,
    output: torch.Tensor | ttnn.Tensor,
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(summaries.shape) != 4 or any(len(tensor.shape) != 3 for tensor in (initial_state, output)):
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in (summaries, initial_state, output)):
        raise ValueError("affine exclusive-scan tensor shapes must be positive")

    batch_heads, key_dim, value_dim = tuple(initial_state.shape)
    slots, transforms, summary_rows, row_width = tuple(summaries.shape)
    if (
        slots not in (1, 2)
        or transforms % batch_heads
        or (summary_rows, row_width) != (key_dim, key_dim + value_dim)
        or tuple(output.shape) != (transforms, key_dim, value_dim)
    ):
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    groups_per_head = transforms // batch_heads

    transitions = batch_heads * (groups_per_head - 1)
    return (
        perf_model.FpuOps(
            matrix_flops=transitions * 2 * key_dim**2 * value_dim,
            add_ops=transitions * key_dim * value_dim,
        ),
        perf_model.SfpuOps(),
    )


def _affine_exclusive_scan_performance(
    summaries: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    output: ttnn.Tensor,
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _affine_exclusive_scan_ops(summaries, initial_state, output)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=(summaries, initial_state),
        outputs=(output,),
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_affine_exclusive_scan_work_golden() -> None:
    fpu, sfpu = _affine_exclusive_scan_ops(
        torch.empty((2, 2, 2, 3)),
        torch.empty((1, 2, 1)),
        torch.empty((2, 2, 1)),
    )

    assert fpu == perf_model.FpuOps(matrix_flops=8, add_ops=2)
    assert sfpu == perf_model.SfpuOps()


def _host_inputs(
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
    *,
    seed: int = 621,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    leading = batch_heads * groups_per_head
    eye = torch.eye(key_dim).reshape(1, key_dim, key_dim)
    a = (0.94 * eye).expand(leading, -1, -1).clone()
    a += 0.0125 * torch.randn(a.shape, generator=generator)
    b = 0.025 * torch.randn(leading, key_dim, value_dim, generator=generator)
    initial_state = 0.05 * torch.randn(batch_heads, key_dim, value_dim, generator=generator)

    # Make adjacent summaries visibly non-commuting so reversed or inclusive scans fail loudly.
    for head in range(batch_heads):
        offset = head * groups_per_head
        if groups_per_head > 1:
            a[offset, 0, 1] += 0.25
            a[offset + 1, 1, 0] -= 0.20
            assert torch.max(torch.abs(a[offset + 1] @ a[offset] - a[offset] @ a[offset + 1])) > 0.01
    return a, b, initial_state


def _oracle(
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    batch_heads: int,
    groups_per_head: int,
) -> torch.Tensor:
    key_dim, value_dim = a.shape[-1], b.shape[-1]
    a = a.reshape(batch_heads, groups_per_head, key_dim, key_dim).float()
    b = b.reshape(batch_heads, groups_per_head, key_dim, value_dim).float()
    entries = []
    for head in range(batch_heads):
        carry = initial_state[head].float()
        for group in range(groups_per_head):
            entries.append(carry.clone())
            carry = a[head, group] @ carry + b[head, group]
    return torch.stack(entries)


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.float32,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _pack(a: torch.Tensor, b: torch.Tensor, *, slots: int = 1) -> torch.Tensor:
    """Packed [slots, N, K, K+V] summaries of [A | B] rows. An unsplit rank never reads a tail slot, so it is NaN."""
    head = torch.cat([a, b], dim=-1)
    return torch.stack([head, torch.full_like(head, float("nan"))][:slots])


def _device_inputs(
    host: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.float32,
    *,
    slots: int = 1,
    placement: str = "interleaved",
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Device summaries and FLOAT32 initial state; ``placement`` is interleaved, nd-sharded-l1 or height-sharded-l1."""
    a, b, initial_state = host
    transforms, key_dim, value_dim = a.shape[0], a.shape[1], b.shape[-1]
    if placement == "interleaved":
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    elif placement == "nd-sharded-l1":
        memory_config = group_summary_nd_sharded(device, slots, transforms, key_dim, value_dim)
    else:
        assert placement == "height-sharded-l1"
        memory_config = height_sharded_memory_config(device, slots * transforms, key_dim, key_dim + value_dim)
    summaries = _to_device(_pack(a, b, slots=slots), device, dtype, memory_config=memory_config)
    return summaries, _to_device(initial_state, device)


def _run(
    summaries: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    groups_per_head: int,
    *,
    actual_start: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> ttnn.Tensor:
    """Run on one device; summaries with a tail slot take initial_state as their (never read) tail seed."""
    tail_entry_states = initial_state if tuple(summaries.shape)[0] == 2 else None
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.affine_exclusive_scan(
            summaries,
            initial_state,
            groups_per_head,
            actual_start=actual_start,
            local_rows=32 * groups_per_head,
            tail_entry_states=tail_entry_states,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )


def _composed_ttnn_baseline(
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    device: ttnn.Device,
    batch_heads: int,
    groups_per_head: int,
) -> ttnn.Tensor:
    """Express the same ordered exclusive scan using ordinary TTNN matmul, add, and concat."""
    key_dim, value_dim = a.shape[-1], b.shape[-1]
    a = a.reshape(batch_heads, groups_per_head, key_dim, key_dim)
    b = b.reshape(batch_heads, groups_per_head, key_dim, value_dim)
    a_groups = [_to_device(a[:, group], device) for group in range(groups_per_head)]
    b_groups = [_to_device(b[:, group], device) for group in range(groups_per_head)]
    carry = _to_device(initial_state, device)
    entries = []
    for group in range(groups_per_head):
        entries.append(carry)
        if group + 1 < groups_per_head:
            carry = ttnn.add(
                ttnn.matmul(a_groups[group], carry, dtype=ttnn.float32),
                b_groups[group],
                dtype=ttnn.float32,
            )
    group_major = ttnn.concat(entries, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    grouped = ttnn.reshape(group_major, [groups_per_head, batch_heads, key_dim, value_dim])
    head_major = ttnn.permute(grouped, (1, 0, 2, 3))
    return ttnn.reshape(head_major, [batch_heads * groups_per_head, key_dim, value_dim])


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "slots,placement",
    [(1, "interleaved"), (2, "interleaved"), (2, "nd-sharded-l1"), (1, "height-sharded-l1")],
    ids=["head-interleaved", "head-tail-interleaved", "head-tail-nd-sharded-l1", "head-height-sharded-l1"],
)
def test_affine_exclusive_scan_contract_and_trace(
    zero_actual_start,
    device: ttnn.Device,
    summary_dtype: ttnn.DataType,
    slots: int,
    placement: str,
) -> None:
    """An unsplit rank scans slot 0; a NaN tail slot must not reach the entry states."""
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 32
    host = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected = _oracle(*host, batch_heads, groups_per_head)
    output_memory = ttnn.DRAM_MEMORY_CONFIG if placement == "interleaved" else ttnn.L1_MEMORY_CONFIG
    summaries_tt, state_tt = _device_inputs(host, device, summary_dtype, slots=slots, placement=placement)
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in (summaries_tt, state_tt))

    first = _run(summaries_tt, state_tt, groups_per_head, memory_config=output_memory, actual_start=zero_actual_start)
    assert first.dtype == ttnn.float32
    assert first.layout == ttnn.TILE_LAYOUT
    assert first.memory_config() == output_memory
    assert tuple(ttnn.to_torch(first).shape) == (batch_heads * groups_per_head, key_dim, value_dim)
    assert first.buffer_address() not in (summaries_tt.buffer_address(), state_tt.buffer_address())

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = _run(summaries_tt, state_tt, groups_per_head, memory_config=output_memory, actual_start=zero_actual_start)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    actual = ttnn.to_torch(first)
    assert_accurate(
        expected, actual, name=f"{summary_dtype} exclusive entry states", pcc_threshold=0.999, nrmse_threshold=0.01
    )
    assert_accurate(
        snapshots[1],
        actual.reshape(batch_heads, groups_per_head, key_dim, value_dim)[:, 0],
        name="exclusive first entry",
        pcc_threshold=0.999,
        nrmse_threshold=0.0006,
    )
    assert_bit_identical(actual, ttnn.to_torch(traced), name="trace replay")
    after = ttnn.to_torch(summaries_tt)
    assert_bit_identical(snapshots[0][0], after[0], name="head slot immutability")
    if slots == 2:
        assert torch.isnan(after[1]).all(), "the NaN tail slot must stay untouched"
    assert_bit_identical(snapshots[1], ttnn.to_torch(state_tt), name="initial_state immutability")
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    ("batch_heads", "groups_per_head", "key_dim", "value_dim", "summary_dtype", "placement"),
    [
        pytest.param(2, 1, 32, 32, ttnn.float32, "interleaved", id="bh2-g1-k32-v32-fp32-interleaved"),
        pytest.param(2, 3, 32, 32, ttnn.bfloat16, "nd-sharded-l1", id="bh2-g3-k32-v32-bf16-nd-sharded-l1"),
        pytest.param(3, 2, 32, 64, ttnn.float32, "height-sharded-l1", id="bh3-g2-k32-v64-fp32-height-sharded-l1"),
    ],
)
def test_affine_exclusive_scan_shape_accuracy(
    zero_actual_start,
    device: ttnn.Device,
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
    summary_dtype: ttnn.DataType,
    placement: str,
) -> None:
    host = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected = _oracle(*host, batch_heads, groups_per_head)
    device_inputs = _device_inputs(host, device, summary_dtype, slots=2, placement=placement)

    output = _run(*device_inputs, groups_per_head, actual_start=zero_actual_start)

    assert_accurate(
        expected,
        ttnn.to_torch(output),
        name=f"{summary_dtype} shape-sweep exclusive entry states",
        pcc_threshold=0.999,
        nrmse_threshold=0.007,
    )


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
def test_affine_exclusive_scan_is_device_deterministic(
    zero_actual_start, device: ttnn.Device, summary_dtype: ttnn.DataType
) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1441)
    device_inputs = _device_inputs(host, device, summary_dtype)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)

    def run() -> tuple[ttnn.Tensor]:
        return (_run(*device_inputs, case.groups_per_head, actual_start=zero_actual_start),)

    (output_tt,), (output,), mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{summary_dtype} exclusive scan device-side exact-value determinism marker",
    )
    assert_accurate(
        expected, output, name=f"{summary_dtype} exclusive entry states", pcc_threshold=0.999, nrmse_threshold=0.01
    )
    ttnn.deallocate(output_tt)


def test_affine_exclusive_scan_cache_hit_rebinds_fresh_tensors(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _UNIT_CASE
    host_a = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1911)
    host_b = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1912)
    device_a = _device_inputs(host_a, device)
    device_b = _device_inputs(host_b, device)

    output_a = _run(*device_a, case.groups_per_head, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(*device_b, case.groups_per_head, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(device_a, device_b, strict=True))
    assert output_a.buffer_address() != output_b.buffer_address()
    expected_a = _oracle(*host_a, case.batch_heads, case.groups_per_head)
    expected_b = _oracle(*host_b, case.batch_heads, case.groups_per_head)
    actual_a = ttnn.to_torch(output_a)
    actual_b = ttnn.to_torch(output_b)
    assert_accurate(expected_a, actual_a, name="cache miss tensors", pcc_threshold=0.999, nrmse_threshold=0.01)
    assert_accurate(expected_b, actual_b, name="cache hit fresh tensors", pcc_threshold=0.999, nrmse_threshold=0.01)
    assert not torch.equal(actual_a, actual_b)


def test_affine_exclusive_scan_default_compute_config_matches_explicit_defaults(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=817)
    device_inputs = _device_inputs(host, device)
    implicit = _run(*device_inputs, case.groups_per_head, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = _run(
        *device_inputs, case.groups_per_head, compute_kernel_config=explicit_config, actual_start=zero_actual_start
    )
    assert device.num_program_cache_entries() == entries
    assert_bit_identical(
        ttnn.to_torch(implicit),
        ttnn.to_torch(explicit),
        name="implicit vs explicit compute defaults",
    )


def test_affine_exclusive_scan_approximate_math_uses_distinct_accurate_program(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=818)
    device_inputs = _device_inputs(host, device)
    exact = _run(*device_inputs, case.groups_per_head, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = _run(
        *device_inputs, case.groups_per_head, compute_kernel_config=approximate_config, actual_start=zero_actual_start
    )
    assert device.num_program_cache_entries() == entries + 1
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    assert_accurate(expected, ttnn.to_torch(exact), name="exact math", pcc_threshold=0.999, nrmse_threshold=0.01)
    assert_accurate(
        expected, ttnn.to_torch(approximate), name="approximate math", pcc_threshold=0.999, nrmse_threshold=0.002
    )


def test_affine_exclusive_scan_rejects_unsupported_compute_config(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim)
    device_inputs = _device_inputs(host, device)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(
            *device_inputs,
            case.groups_per_head,
            compute_kernel_config=unsupported_config,
            actual_start=zero_actual_start,
        )


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("placement", ["dram-head", "nd-sharded-l1-head-tail"])
@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_affine_exclusive_scan_production_performance(
    zero_actual_start, device: ttnn.Device, case: _ProductionCase, placement: str
) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for affine exclusive-scan performance checks")

    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=117)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    if placement == "dram-head":
        device_inputs = _device_inputs(host, device, ttnn.bfloat16)
        expected_duration_ns = case.expected_duration_ns
    else:
        device_inputs = _device_inputs(host, device, ttnn.bfloat16, slots=2, placement="nd-sharded-l1")
        expected_duration_ns = _ND_SHARDED_DURATION_NS[case.case_id]
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )

    def run() -> ttnn.Tensor:
        return _run(
            *device_inputs,
            case.groups_per_head,
            compute_kernel_config=compute_kernel_config,
            actual_start=zero_actual_start,
        )

    output, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(output.shape) == (
        case.batch_heads * case.groups_per_head,
        case.key_dim,
        case.value_dim,
    )
    assert output.dtype == ttnn.float32
    assert_accurate(
        expected,
        ttnn.to_torch(output),
        name=f"{case.case_id} production output",
        pcc_threshold=0.999,
        nrmse_threshold=0.02,
    )
    performance = _affine_exclusive_scan_performance(
        *device_inputs,
        output,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    logger.info(
        f"affine exclusive scan {case.case_id} {placement}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    lower = expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
    upper = expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} {placement} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


def test_affine_exclusive_scan_matches_composed_ttnn_baseline(zero_actual_start, device: ttnn.Device) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 64
    a, b, initial_state = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim, seed=812)
    expected = _oracle(a, b, initial_state, batch_heads, groups_per_head)
    fused = _run(*_device_inputs((a, b, initial_state), device), groups_per_head, actual_start=zero_actual_start)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        composed = _composed_ttnn_baseline(a, b, initial_state, device, batch_heads, groups_per_head)
    ttnn.synchronize_device(device)
    assert_accurate(
        expected, ttnn.to_torch(fused), name="fused exclusive scan", pcc_threshold=0.999, nrmse_threshold=0.01
    )
    assert_accurate(
        expected,
        ttnn.to_torch(composed),
        name="composed TTNN exclusive scan",
        pcc_threshold=0.999,
        nrmse_threshold=0.002,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host", "summaries must be an allocated device tensor"),
        ("host_state", "initial_state must be an allocated device tensor"),
        ("dtype", "summaries must be FLOAT32 or BFLOAT16"),
        ("state_dtype", "initial_state must be FLOAT32"),
        ("layout", "summaries must use TILE layout"),
        ("rank", "summaries must be rank 4"),
        ("slots", "summaries must hold 1 or 2 slots"),
        ("nondivisible", "transforms per slot must be divisible by groups_per_head"),
        ("no_offset", "summary rows must be"),
        ("unaligned", "positive, tile-aligned K and V"),
        ("state_shape", "initial_state shape must be"),
        ("tail_without_seed", "tail_entry_states is required exactly when the summaries hold a tail slot"),
        ("seed_without_tail", "tail_entry_states is required exactly when the summaries hold a tail slot"),
        ("tail_seed_shape", "tail_entry_states shape must match initial_state"),
    ],
)
def test_affine_exclusive_scan_rejects_invalid_inputs(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    a, b, initial_state = _host_inputs(1, 4, 32, 32)
    groups_per_head = 4
    packed = _pack(a, b)
    summaries_tt = _to_device(packed, device)
    state_tt = _to_device(initial_state, device)
    tail_entry_states = None

    if case == "host":
        summaries_tt = ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "host_state":
        state_tt = ttnn.from_torch(initial_state, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "dtype":
        summaries_tt = _to_device(packed, device, ttnn.bfloat8_b)
    elif case == "state_dtype":
        state_tt = _to_device(initial_state, device, ttnn.bfloat16)
    elif case == "layout":
        summaries_tt = _to_device(packed, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        summaries_tt = _to_device(packed[0], device)
    elif case == "slots":
        summaries_tt = _to_device(packed.expand(3, -1, -1, -1).contiguous(), device)
    elif case == "nondivisible":
        summaries_tt = _to_device(packed[:, :-1], device)
    elif case == "no_offset":
        summaries_tt = _to_device(packed[..., :32], device)
    elif case == "unaligned":
        summaries_tt = _to_device(packed[..., :63], device)
    elif case == "state_shape":
        state_tt = _to_device(initial_state[:, :, :31], device)
    elif case == "tail_without_seed":
        summaries_tt = _to_device(_pack(a, b, slots=2), device)
    elif case == "seed_without_tail":
        tail_entry_states = state_tt
    elif case == "tail_seed_shape":
        summaries_tt = _to_device(_pack(a, b, slots=2), device)
        tail_entry_states = _to_device(initial_state[:, :, :31], device)
    with expect_error(RuntimeError, message):
        with ttnn.manage_config("throw_exception_on_fallback", True):
            ttnn.experimental.kda.affine_exclusive_scan(
                summaries_tt,
                state_tt,
                groups_per_head,
                actual_start=zero_actual_start,
                local_rows=32 * groups_per_head,
                tail_entry_states=tail_entry_states,
            )


def test_affine_exclusive_scan_rejects_excess_workers(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    grid = device.compute_with_storage_grid_size()
    worker_limit = min(grid.x * grid.y, 128)
    group_workers = worker_limit + 1
    host = _host_inputs(1, group_workers, 32, 32)

    with expect_error(RuntimeError, f"supports at most {worker_limit} group workers on this device"):
        _run(*_device_inputs(host, device), group_workers, actual_start=zero_actual_start)


@pytest.mark.parametrize(
    "memory_layout",
    [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
    ids=["width_sharded", "block_sharded"],
)
def test_affine_exclusive_scan_rejects_unsupported_input_sharding(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
    memory_layout: ttnn.TensorMemoryLayout,
) -> None:
    a, b, initial_state = _host_inputs(1, 4, 32, 32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [128, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    unsupported = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    with expect_error(RuntimeError, "summaries must use interleaved, height-sharded or ND-sharded memory"):
        _run(
            _to_device(_pack(a, b), device, memory_config=unsupported),
            _to_device(initial_state, device),
            4,
            actual_start=zero_actual_start,
        )


def test_affine_exclusive_scan_rejects_invalid_configuration(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    summaries_tt, state_tt = _device_inputs(_host_inputs(1, 4, 32, 32), device)
    with expect_error(RuntimeError, "groups_per_head must be positive"):
        _run(summaries_tt, state_tt, 0, actual_start=zero_actual_start)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [128, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED"):
        _run(summaries_tt, state_tt, 4, memory_config=sharded, actual_start=zero_actual_start)
