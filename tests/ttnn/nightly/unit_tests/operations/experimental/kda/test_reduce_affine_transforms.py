# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA affine-transform reduction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    group_summary_nd_sharded,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    collect_accuracy_and_determinism_results,
    assert_equal,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]


@dataclass(frozen=True)
class _Case:
    case_id: str
    batch_heads: int
    groups_per_head: int
    key_dim: int
    value_dim: int


_PRODUCTION_PERF_MARGIN = 0.05
_SMALL_CASE = _Case(
    "bh2-g4-k32-v64",
    batch_heads=2,
    groups_per_head=4,
    key_dim=32,
    value_dim=64,
)
_PRODUCTION_PERF_CASE = _Case(
    "sp2-tp4-bh24-g4-k128-v128",
    batch_heads=24,
    groups_per_head=4,
    key_dim=128,
    value_dim=128,
)
# Keyed by summary placement: main's DRAM head-only input, and the grouped model's target L1 head+tail shards.
_PRODUCTION_PERF_EXPECTED_DURATION_NS = {"dram-head": 46_120, "nd-sharded-l1-head-tail": 41_950}


def _reduce_affine_transforms_ops(
    summaries: torch.Tensor | ttnn.Tensor,
    outputs: tuple[torch.Tensor | ttnn.Tensor, ...],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(outputs) != 2:
        raise ValueError("affine-transform reduction requires two outputs")
    if len(summaries.shape) != 4 or any(len(tensor.shape) != 3 for tensor in outputs):
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in (summaries, *outputs)):
        raise ValueError("affine-transform reduction tensor shapes must be positive")

    batch_heads, key_dim, output_key_dim = tuple(outputs[0].shape)
    value_dim = tuple(outputs[1].shape)[-1]
    slots, transforms, summary_rows, row_width = tuple(summaries.shape)
    if (
        key_dim != output_key_dim
        or tuple(outputs[1].shape) != (batch_heads, key_dim, value_dim)
        or slots not in (1, 2)
        or transforms % batch_heads
        or (summary_rows, row_width) != (key_dim, key_dim + value_dim)
    ):
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    groups_per_head = transforms // batch_heads

    compositions = batch_heads * (groups_per_head - 1)
    return (
        perf_model.FpuOps(
            matrix_flops=compositions * (2 * key_dim**3 + 2 * key_dim**2 * value_dim),
            add_ops=compositions * key_dim * value_dim,
        ),
        perf_model.SfpuOps(),
    )


def _reduce_affine_transforms_performance(
    summaries: ttnn.Tensor,
    outputs: tuple[ttnn.Tensor, ...],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _reduce_affine_transforms_ops(summaries, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=(summaries,),
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_reduce_affine_transforms_work_golden() -> None:
    fpu, sfpu = _reduce_affine_transforms_ops(
        torch.empty((2, 2, 2, 3)),
        (torch.empty((1, 2, 2)), torch.empty((1, 2, 1))),
    )

    assert fpu == perf_model.FpuOps(matrix_flops=24, add_ops=2)
    assert sfpu == perf_model.SfpuOps()


def _host_inputs(
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
    *,
    seed: int = 914,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    leading = batch_heads * groups_per_head
    eye = torch.eye(key_dim).reshape(1, key_dim, key_dim)
    a = (0.94 * eye).expand(leading, -1, -1).clone()
    a += 0.0125 * torch.randn(a.shape, generator=generator)
    b = 0.025 * torch.randn(leading, key_dim, value_dim, generator=generator)

    # Make the first pair visibly non-commuting so an accidentally reversed reduction fails loudly.
    for head in range(batch_heads):
        offset = head * groups_per_head
        if groups_per_head > 1:
            a[offset, 0, 1] += 0.25
            a[offset + 1, 1, 0] -= 0.20
            assert torch.max(torch.abs(a[offset + 1] @ a[offset] - a[offset] @ a[offset + 1])) > 0.01
    return a, b


def _oracle(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_heads: int,
    groups_per_head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_dim, value_dim = a.shape[-1], b.shape[-1]
    a = a.reshape(batch_heads, groups_per_head, key_dim, key_dim).float()
    b = b.reshape(batch_heads, groups_per_head, key_dim, value_dim).float()
    reduced_a = []
    reduced_b = []
    for head in range(batch_heads):
        total_a = torch.eye(key_dim)
        total_b = torch.zeros(key_dim, value_dim)
        for group in range(groups_per_head):
            total_a = a[head, group] @ total_a
            total_b = a[head, group] @ total_b + b[head, group]
        reduced_a.append(total_a)
        reduced_b.append(total_b)
    return torch.stack(reduced_a), torch.stack(reduced_b)


def _pack(a: torch.Tensor, b: torch.Tensor, *, slots: int = 1) -> torch.Tensor:
    """Packed [slots, N, K, K+V] summaries of [A | B] rows. A tail slot is NaN: the reduction must never read it."""
    head = torch.cat([a, b], dim=-1)
    return torch.stack([head, torch.full_like(head, float("nan"))][:slots])


def _summary_memory_config(
    placement: str, device: ttnn.Device, slots: int, transforms: int, key_dim: int, value_dim: int
) -> ttnn.MemoryConfig:
    if placement == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    if placement == "nd-sharded-l1":
        return group_summary_nd_sharded(device, slots, transforms, key_dim, value_dim)
    assert placement == "height-sharded-l1"
    # One [K, K+V] transform per core, slot-major.
    return _height_sharded_memory_config(device, slots * transforms, key_dim, key_dim + value_dim)


def _device_summaries(
    a: torch.Tensor,
    b: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.float32,
    *,
    slots: int = 1,
    placement: str = "interleaved",
) -> ttnn.Tensor:
    transforms, key_dim, value_dim = a.shape[0], a.shape[1], b.shape[-1]
    memory_config = _summary_memory_config(placement, device, slots, transforms, key_dim, value_dim)
    return _to_device(_pack(a, b, slots=slots), device, dtype, memory_config=memory_config)


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.float32,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _height_sharded_memory_config(
    device: ttnn.Device, leading: int, matrix_height: int, matrix_width: int
) -> ttnn.MemoryConfig:
    cores = ttnn.num_cores_to_corerangeset(leading, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.create_sharded_memory_config(
        (leading, matrix_height, matrix_width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _run(
    summaries: ttnn.Tensor,
    groups_per_head: int,
    *,
    actual_start: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.reduce_affine_transforms(
            summaries,
            groups_per_head,
            actual_start=actual_start,
            local_rows=32 * groups_per_head,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )


def _composed_ttnn_baseline(
    a: torch.Tensor,
    b: torch.Tensor,
    device: ttnn.Device,
    batch_heads: int,
    groups_per_head: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Express the same ordered reduction using only ordinary TTNN matmul and add."""
    key_dim, value_dim = a.shape[-1], b.shape[-1]
    a = a.reshape(batch_heads, groups_per_head, key_dim, key_dim)
    b = b.reshape(batch_heads, groups_per_head, key_dim, value_dim)
    a_groups = [_to_device(a[:, group], device) for group in range(groups_per_head)]
    b_groups = [_to_device(b[:, group], device) for group in range(groups_per_head)]
    total_a = a_groups[0]
    total_b = b_groups[0]
    for group in range(1, groups_per_head):
        total_b = ttnn.add(
            ttnn.matmul(a_groups[group], total_b, dtype=ttnn.float32),
            b_groups[group],
            dtype=ttnn.float32,
        )
        total_a = ttnn.matmul(a_groups[group], total_a, dtype=ttnn.float32)
    return total_a, total_b


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "slots,placement",
    [(1, "interleaved"), (2, "interleaved"), (2, "nd-sharded-l1"), (1, "height-sharded-l1")],
    ids=["head-interleaved", "head-tail-interleaved", "head-tail-nd-sharded-l1", "head-height-sharded-l1"],
)
def test_reduce_affine_transforms_contract_and_trace(
    zero_actual_start,
    device: ttnn.Device,
    summary_dtype: ttnn.DataType,
    slots: int,
    placement: str,
) -> None:
    """Slot 0 is reduced; a NaN tail slot must not reach the outputs."""
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 32
    a, b = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected_a, expected_b = _oracle(a, b, batch_heads, groups_per_head)
    summaries_tt = _device_summaries(a, b, device, summary_dtype, slots=slots, placement=placement)
    assert summaries_tt.memory_config() == _summary_memory_config(
        placement, device, slots, batch_heads * groups_per_head, key_dim, value_dim
    )
    snapshot = ttnn.to_torch(summaries_tt).clone()

    first = _run(summaries_tt, groups_per_head, actual_start=zero_actual_start)
    for output, shape in zip(first, ((batch_heads, key_dim, key_dim), (batch_heads, key_dim, value_dim)), strict=True):
        assert output.dtype == ttnn.float32
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(ttnn.to_torch(output).shape) == shape
        assert output.buffer_address() != summaries_tt.buffer_address()

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = _run(summaries_tt, groups_per_head, actual_start=zero_actual_start)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, golden, actual_tt, traced_tt in zip(("A", "B"), (expected_a, expected_b), first, traced, strict=True):
        actual = ttnn.to_torch(actual_tt)
        assert_accurate(
            golden, actual, name=f"{summary_dtype} reduced {name}", pcc_threshold=0.999, nrmse_threshold=0.02
        )
        assert_bit_identical(actual, ttnn.to_torch(traced_tt), name=f"{name} trace replay")

    after = ttnn.to_torch(summaries_tt)
    assert_bit_identical(snapshot[0], after[0], name="head slot immutability")
    if slots == 2:
        assert torch.isnan(after[1]).all(), "the NaN tail slot must stay untouched"
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    ("batch_heads", "groups_per_head", "key_dim", "value_dim", "summary_dtype", "placement"),
    [
        pytest.param(2, 1, 32, 32, ttnn.float32, "interleaved", id="bh2-g1-k32-v32-fp32-interleaved"),
        pytest.param(2, 3, 32, 32, ttnn.bfloat16, "nd-sharded-l1", id="bh2-g3-k32-v32-bf16-nd-sharded-l1"),
        pytest.param(3, 2, 32, 64, ttnn.float32, "interleaved", id="bh3-g2-k32-v64-fp32-interleaved"),
        pytest.param(2, 2, 160, 32, ttnn.bfloat16, "nd-sharded-l1", id="bh2-g2-k160-v32-bf16-nd-sharded-l1"),
        pytest.param(2, 2, 32, 160, ttnn.float32, "height-sharded-l1", id="bh2-g2-k32-v160-fp32-height-sharded-l1"),
    ],
)
def test_reduce_affine_transforms_shape_accuracy(
    zero_actual_start,
    device: ttnn.Device,
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
    summary_dtype: ttnn.DataType,
    placement: str,
) -> None:
    a, b = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected = _oracle(a, b, batch_heads, groups_per_head)
    summaries_tt = _device_summaries(a, b, device, summary_dtype, slots=2, placement=placement)

    outputs = _run(summaries_tt, groups_per_head, actual_start=zero_actual_start)

    for name, golden, output in zip(("A", "B"), expected, outputs, strict=True):
        assert_accurate(
            golden,
            ttnn.to_torch(output),
            name=f"{summary_dtype} shape-sweep reduced {name}",
            pcc_threshold=0.999,
            nrmse_threshold=0.01,
        )


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
def test_reduce_affine_transforms_is_device_deterministic(
    zero_actual_start, device: ttnn.Device, summary_dtype: ttnn.DataType
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1441)
    summaries_tt = _device_summaries(*host, device, summary_dtype)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)

    def run() -> tuple[ttnn.Tensor, ...]:
        return _run(summaries_tt, case.groups_per_head, actual_start=zero_actual_start)

    reference_outputs, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{summary_dtype} reduced outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("A", "B"), expected, outputs, strict=True):
        assert_accurate(
            golden, output, name=f"{summary_dtype} reduced {name}", pcc_threshold=0.999, nrmse_threshold=0.02
        )
    for output in reference_outputs:
        ttnn.deallocate(output)


def test_reduce_affine_transforms_cache_hit_rebinds_fresh_tensors(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host_a = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1911)
    host_b = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1912)
    device_a = _device_summaries(*host_a, device)
    device_b = _device_summaries(*host_b, device)

    output_a = _run(device_a, case.groups_per_head, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(device_b, case.groups_per_head, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert device_a.buffer_address() != device_b.buffer_address()
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(output_a, output_b, strict=True))
    expected_a = _oracle(*host_a, case.batch_heads, case.groups_per_head)
    expected_b = _oracle(*host_b, case.batch_heads, case.groups_per_head)
    for name, golden_a, golden_b, actual_a_tt, actual_b_tt in zip(
        ("A", "B"), expected_a, expected_b, output_a, output_b, strict=True
    ):
        actual_a = ttnn.to_torch(actual_a_tt)
        actual_b = ttnn.to_torch(actual_b_tt)
        assert_accurate(
            golden_a, actual_a, name=f"{name} cache miss tensors", pcc_threshold=0.999, nrmse_threshold=0.02
        )
        assert_accurate(
            golden_b, actual_b, name=f"{name} cache hit fresh tensors", pcc_threshold=0.999, nrmse_threshold=0.02
        )
        assert not torch.equal(actual_a, actual_b)


def test_reduce_affine_transforms_default_compute_config_matches_explicit_defaults(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=817)
    summaries_tt = _device_summaries(*host, device)
    implicit = _run(summaries_tt, case.groups_per_head, actual_start=zero_actual_start)
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
        summaries_tt, case.groups_per_head, compute_kernel_config=explicit_config, actual_start=zero_actual_start
    )
    assert device.num_program_cache_entries() == entries
    for name, implicit_output, explicit_output in zip(("A", "B"), implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_output),
            ttnn.to_torch(explicit_output),
            name=f"{name} implicit vs explicit compute defaults",
        )


def test_reduce_affine_transforms_approximate_math_uses_distinct_accurate_program(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=818)
    summaries_tt = _device_summaries(*host, device)
    exact = _run(summaries_tt, case.groups_per_head, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = _run(
        summaries_tt, case.groups_per_head, compute_kernel_config=approximate_config, actual_start=zero_actual_start
    )
    assert device.num_program_cache_entries() == entries + 1
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    for name, golden, exact_output, approximate_output in zip(("A", "B"), expected, exact, approximate, strict=True):
        assert_accurate(
            golden, ttnn.to_torch(exact_output), name=f"{name} exact math", pcc_threshold=0.999, nrmse_threshold=0.02
        )
        assert_accurate(
            golden,
            ttnn.to_torch(approximate_output),
            name=f"{name} approximate math",
            pcc_threshold=0.999,
            nrmse_threshold=0.003,
        )


def test_reduce_affine_transforms_rejects_unsupported_compute_config(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim)
    summaries_tt = _device_summaries(*host, device)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(
            summaries_tt, case.groups_per_head, compute_kernel_config=unsupported_config, actual_start=zero_actual_start
        )


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("placement", ["dram-head", "nd-sharded-l1-head-tail"])
def test_reduce_affine_transforms_production_performance(
    zero_actual_start, device: ttnn.Device, placement: str
) -> None:
    case = _PRODUCTION_PERF_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for affine-transform reduction performance checks")

    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=117)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    if placement == "dram-head":
        summaries_tt = _device_summaries(*host, device, ttnn.bfloat16)
    else:
        # The target placement of the sequence-parallel grouped summary: BF16 head and tail slots, one ND shard per
        # group-head on the core its reduction worker occupies.
        summaries_tt = _device_summaries(*host, device, ttnn.bfloat16, slots=2, placement="nd-sharded-l1")
    production_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return _run(
            summaries_tt,
            case.groups_per_head,
            compute_kernel_config=production_compute_config,
            actual_start=zero_actual_start,
        )

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(tuple(output.shape) for output in outputs) == (
        (case.batch_heads, case.key_dim, case.key_dim),
        (case.batch_heads, case.key_dim, case.value_dim),
    )
    assert all(output.dtype == ttnn.float32 for output in outputs)
    for name, golden, output in zip(("A", "B"), expected, outputs, strict=True):
        assert_accurate(
            golden, ttnn.to_torch(output), name=f"production reduced {name}", pcc_threshold=0.999, nrmse_threshold=0.015
        )
    performance = _reduce_affine_transforms_performance(
        summaries_tt,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    logger.info(
        f"affine-transform reduction {case.case_id} {placement}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    reference_ns = _PRODUCTION_PERF_EXPECTED_DURATION_NS[placement]
    lower = reference_ns * (1 - _PRODUCTION_PERF_MARGIN)
    upper = reference_ns * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} {placement} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {reference_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


def test_reduce_affine_transforms_matches_composed_ttnn_baseline(zero_actual_start, device: ttnn.Device) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 64
    a, b = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim, seed=117)
    expected = _oracle(a, b, batch_heads, groups_per_head)
    fused = _run(_device_summaries(a, b, device), groups_per_head, actual_start=zero_actual_start)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        composed = _composed_ttnn_baseline(a, b, device, batch_heads, groups_per_head)
    ttnn.synchronize_device(device)
    for name, golden, fused_tt, composed_tt in zip(("A", "B"), expected, fused, composed, strict=True):
        assert_accurate(
            golden, ttnn.to_torch(fused_tt), name=f"fused {name}", pcc_threshold=0.999, nrmse_threshold=0.02
        )
        assert_accurate(
            golden, ttnn.to_torch(composed_tt), name=f"composed TTNN {name}", pcc_threshold=0.999, nrmse_threshold=0.004
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host", "summaries must be an allocated device tensor"),
        ("dtype", "summaries must be FLOAT32 or BFLOAT16"),
        ("layout", "summaries must use TILE layout"),
        ("rank", "summaries must be rank 4"),
        ("slots", "summaries must hold 1 or 2 slots"),
        ("nondivisible", "transforms per slot must be divisible by groups_per_head"),
        ("no_offset", "summary rows must be"),
        ("unaligned", "positive, tile-aligned K and V"),
    ],
)
def test_reduce_affine_transforms_rejects_invalid_inputs(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    a, b = _host_inputs(1, 4, 32, 32)
    groups_per_head = 4
    packed = _pack(a, b)
    summaries_tt = _to_device(packed, device)

    if case == "host":
        summaries_tt = ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "dtype":
        summaries_tt = _to_device(packed, device, ttnn.bfloat8_b)
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
    with expect_error(RuntimeError, message):
        _run(summaries_tt, groups_per_head, actual_start=zero_actual_start)


def test_reduce_affine_transforms_rejects_excess_workers(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    grid = device.compute_with_storage_grid_size()
    worker_limit = min(grid.x * grid.y, 128)
    group_workers = worker_limit + 1
    a, b = _host_inputs(1, group_workers, 32, 32)

    with expect_error(RuntimeError, f"supports at most {worker_limit} group workers on this device"):
        _run(_device_summaries(a, b, device), group_workers, actual_start=zero_actual_start)


@pytest.mark.parametrize(
    "memory_layout",
    [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
    ids=["width_sharded", "block_sharded"],
)
def test_reduce_affine_transforms_rejects_unsupported_input_sharding(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
    memory_layout: ttnn.TensorMemoryLayout,
) -> None:
    a, b = _host_inputs(1, 4, 32, 32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [128, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    unsupported = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)
    summaries_tt = _to_device(_pack(a, b), device, memory_config=unsupported)

    with expect_error(RuntimeError, "summaries must use interleaved, height-sharded or ND-sharded memory"):
        _run(summaries_tt, 4, actual_start=zero_actual_start)


def test_reduce_affine_transforms_rejects_invalid_configuration(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    a, b = _host_inputs(1, 4, 32, 32)
    summaries_tt = _device_summaries(a, b, device)
    with expect_error(RuntimeError, "groups_per_head must be positive"):
        _run(summaries_tt, 0, actual_start=zero_actual_start)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [32, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"):
        _run(summaries_tt, 4, memory_config=sharded, actual_start=zero_actual_start)
