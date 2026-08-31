# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for the experimental KDA affine exclusive scan."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    collect_accuracy_and_determinism_results,
    assert_equal,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 2_000_000}),
]


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    batch_heads: int
    groups_per_head: int
    key_dim: int
    value_dim: int
    expected_duration_ns: int | None


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

# Kimi-K3 production layouts. References are pooled medians from two independent
# three-sample Blackhole runs after scoping synchronization to each independent head.
_PRODUCTION_CASES = (
    _ProductionCase("sp1-tp8", 12, 8, 128, 128, 96782),
    _ProductionCase("sp2-tp4", 24, 4, 128, 128, 76542),
    _ProductionCase("sp4-tp2", 48, 2, 128, 128, 66513),
)


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
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    groups_per_head: int,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> ttnn.Tensor:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.affine_exclusive_scan(
            a,
            b,
            initial_state,
            groups_per_head,
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
@pytest.mark.parametrize("sharded_inputs", [False, True], ids=("interleaved", "height-sharded-l1"))
def test_affine_exclusive_scan_contract_and_trace(
    device: ttnn.Device,
    summary_dtype: ttnn.DataType,
    sharded_inputs: bool,
) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 32
    a, b, initial_state = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected = _oracle(a, b, initial_state, batch_heads, groups_per_head)
    leading = batch_heads * groups_per_head
    a_memory = (
        _height_sharded_memory_config(device, leading, key_dim, key_dim) if sharded_inputs else ttnn.DRAM_MEMORY_CONFIG
    )
    b_memory = (
        _height_sharded_memory_config(device, leading, key_dim, value_dim)
        if sharded_inputs
        else ttnn.DRAM_MEMORY_CONFIG
    )
    output_memory = ttnn.L1_MEMORY_CONFIG if sharded_inputs else ttnn.DRAM_MEMORY_CONFIG
    a_tt = _to_device(a, device, summary_dtype, memory_config=a_memory)
    b_tt = _to_device(b, device, summary_dtype, memory_config=b_memory)
    state_tt = _to_device(initial_state, device)
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in (a_tt, b_tt, state_tt))

    first = _run(a_tt, b_tt, state_tt, groups_per_head, memory_config=output_memory)
    assert first.dtype == ttnn.float32
    assert first.layout == ttnn.TILE_LAYOUT
    assert first.memory_config() == output_memory
    assert tuple(ttnn.to_torch(first).shape) == (batch_heads * groups_per_head, key_dim, value_dim)
    assert first.buffer_address() not in (a_tt.buffer_address(), b_tt.buffer_address(), state_tt.buffer_address())

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = _run(a_tt, b_tt, state_tt, groups_per_head, memory_config=output_memory)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    actual = ttnn.to_torch(first)
    assert_accurate(expected, actual, name=f"{summary_dtype} exclusive entry states", pcc_threshold=0.999)
    assert_accurate(
        snapshots[2],
        actual.reshape(batch_heads, groups_per_head, key_dim, value_dim)[:, 0],
        name="exclusive first entry",
        pcc_threshold=0.999,
    )
    assert_bit_identical(actual, ttnn.to_torch(traced), name="trace replay")
    for name, snapshot, tensor in zip(("a", "b", "initial_state"), snapshots, (a_tt, b_tt, state_tt), strict=True):
        assert_bit_identical(snapshot, ttnn.to_torch(tensor), name=f"{name} immutability")
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    ("batch_heads", "groups_per_head", "key_dim", "value_dim", "summary_dtype", "sharded_inputs"),
    [
        pytest.param(2, 1, 32, 32, ttnn.float32, False, id="bh2-g1-k32-v32-fp32-interleaved"),
        pytest.param(2, 3, 32, 32, ttnn.bfloat16, True, id="bh2-g3-k32-v32-bf16-height-sharded-l1"),
        pytest.param(3, 2, 32, 64, ttnn.float32, False, id="bh3-g2-k32-v64-fp32-interleaved"),
    ],
)
def test_affine_exclusive_scan_shape_accuracy(
    device: ttnn.Device,
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
    summary_dtype: ttnn.DataType,
    sharded_inputs: bool,
) -> None:
    a, b, initial_state = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected = _oracle(a, b, initial_state, batch_heads, groups_per_head)
    leading = batch_heads * groups_per_head
    a_memory = (
        _height_sharded_memory_config(device, leading, key_dim, key_dim) if sharded_inputs else ttnn.DRAM_MEMORY_CONFIG
    )
    b_memory = (
        _height_sharded_memory_config(device, leading, key_dim, value_dim)
        if sharded_inputs
        else ttnn.DRAM_MEMORY_CONFIG
    )
    device_inputs = (
        _to_device(a, device, summary_dtype, memory_config=a_memory),
        _to_device(b, device, summary_dtype, memory_config=b_memory),
        _to_device(initial_state, device),
    )

    output = _run(*device_inputs, groups_per_head)

    assert_accurate(
        expected,
        ttnn.to_torch(output),
        name=f"{summary_dtype} shape-sweep exclusive entry states",
        pcc_threshold=0.999,
    )


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
def test_affine_exclusive_scan_is_device_deterministic(device: ttnn.Device, summary_dtype: ttnn.DataType) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1441)
    device_inputs = (
        _to_device(host[0], device, summary_dtype),
        _to_device(host[1], device, summary_dtype),
        _to_device(host[2], device),
    )
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)

    def run() -> tuple[ttnn.Tensor]:
        return (_run(*device_inputs, case.groups_per_head),)

    (output_tt,), (output,), mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{summary_dtype} exclusive scan device-side exact-value determinism marker",
    )
    assert_accurate(expected, output, name=f"{summary_dtype} exclusive entry states", pcc_threshold=0.999)
    ttnn.deallocate(output_tt)


def test_affine_exclusive_scan_cache_hit_rebinds_fresh_tensors(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _UNIT_CASE
    host_a = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1911)
    host_b = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1912)
    device_a = (
        _to_device(host_a[0], device),
        _to_device(host_a[1], device),
        _to_device(host_a[2], device),
    )
    device_b = (
        _to_device(host_b[0], device),
        _to_device(host_b[1], device),
        _to_device(host_b[2], device),
    )

    output_a = _run(*device_a, case.groups_per_head)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(*device_b, case.groups_per_head)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(device_a, device_b, strict=True))
    assert output_a.buffer_address() != output_b.buffer_address()
    expected_a = _oracle(*host_a, case.batch_heads, case.groups_per_head)
    expected_b = _oracle(*host_b, case.batch_heads, case.groups_per_head)
    actual_a = ttnn.to_torch(output_a)
    actual_b = ttnn.to_torch(output_b)
    assert_accurate(expected_a, actual_a, name="cache miss tensors", pcc_threshold=0.999)
    assert_accurate(expected_b, actual_b, name="cache hit fresh tensors", pcc_threshold=0.999)
    assert not torch.equal(actual_a, actual_b)


def test_affine_exclusive_scan_default_compute_config_matches_explicit_defaults(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=817)
    device_inputs = tuple(_to_device(tensor, device) for tensor in host)
    implicit = _run(*device_inputs, case.groups_per_head)
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
    explicit = _run(*device_inputs, case.groups_per_head, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    assert_bit_identical(
        ttnn.to_torch(implicit),
        ttnn.to_torch(explicit),
        name="implicit vs explicit compute defaults",
    )


def test_affine_exclusive_scan_approximate_math_uses_distinct_accurate_program(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=818)
    device_inputs = tuple(_to_device(tensor, device) for tensor in host)
    exact = _run(*device_inputs, case.groups_per_head)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = _run(*device_inputs, case.groups_per_head, compute_kernel_config=approximate_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    assert_accurate(expected, ttnn.to_torch(exact), name="exact math", pcc_threshold=0.999)
    assert_accurate(expected, ttnn.to_torch(approximate), name="approximate math", pcc_threshold=0.999)


def test_affine_exclusive_scan_rejects_unsupported_compute_config(device: ttnn.Device, expect_error: Callable) -> None:
    case = _UNIT_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim)
    device_inputs = tuple(_to_device(tensor, device) for tensor in host)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(*device_inputs, case.groups_per_head, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_affine_exclusive_scan_production_performance(device: ttnn.Device, case: _ProductionCase) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for affine exclusive-scan performance checks")

    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=117)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    device_inputs = (
        _to_device(host[0], device, ttnn.bfloat16),
        _to_device(host[1], device, ttnn.bfloat16),
        _to_device(host[2], device, ttnn.float32),
    )
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
        return _run(*device_inputs, case.groups_per_head, compute_kernel_config=compute_kernel_config)

    output, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(output.shape) == (
        case.batch_heads * case.groups_per_head,
        case.key_dim,
        case.value_dim,
    )
    assert output.dtype == ttnn.float32
    assert_accurate(expected, ttnn.to_torch(output), name=f"{case.case_id} production output", pcc_threshold=0.999)
    logger.info(
        f"affine exclusive scan {case.case_id}: duration={duration_ns:.0f} ns, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )
    if case.expected_duration_ns is not None:
        lower = case.expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
        upper = case.expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
        assert lower <= duration_ns <= upper, (
            f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
            f"(reference {case.expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
        )


def test_affine_exclusive_scan_matches_composed_ttnn_baseline(device: ttnn.Device) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 64
    a, b, initial_state = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim, seed=812)
    expected = _oracle(a, b, initial_state, batch_heads, groups_per_head)
    fused = _run(_to_device(a, device), _to_device(b, device), _to_device(initial_state, device), groups_per_head)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        composed = _composed_ttnn_baseline(a, b, initial_state, device, batch_heads, groups_per_head)
    ttnn.synchronize_device(device)
    assert_accurate(expected, ttnn.to_torch(fused), name="fused exclusive scan", pcc_threshold=0.999)
    assert_accurate(expected, ttnn.to_torch(composed), name="composed TTNN exclusive scan", pcc_threshold=0.999)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_a", "a must be an allocated device tensor"),
        ("host_b", "b must be an allocated device tensor"),
        ("host_state", "initial_state must be an allocated device tensor"),
        ("dtype", "a and b must have matching dtypes"),
        ("state_dtype", "initial_state must be FLOAT32"),
        ("layout", "b must use TILE layout"),
        ("rank", "inputs must be rank 3"),
        ("leading", "matching leading dimensions"),
        ("nondivisible", "leading dimension must be divisible"),
        ("nonsquare", "a must contain square"),
        ("key_dim", "matching K dimensions"),
        ("unaligned", "K and V must be positive and tile aligned"),
        ("state_shape", "initial_state shape must be"),
    ],
)
def test_affine_exclusive_scan_rejects_invalid_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    a, b, initial_state = _host_inputs(1, 4, 32, 32)
    groups_per_head = 4
    a_tt = _to_device(a, device)
    b_tt = _to_device(b, device)
    state_tt = _to_device(initial_state, device)

    if case == "host_a":
        a_tt = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "host_b":
        b_tt = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "host_state":
        state_tt = ttnn.from_torch(initial_state, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "dtype":
        b_tt = _to_device(b, device, ttnn.bfloat16)
    elif case == "state_dtype":
        state_tt = _to_device(initial_state, device, ttnn.bfloat16)
    elif case == "layout":
        b_tt = _to_device(b, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        b_tt = _to_device(b.reshape(1, 4, 32, 32), device)
    elif case == "leading":
        b_tt = _to_device(b[:-1], device)
    elif case == "nondivisible":
        a_tt = _to_device(a[:-1], device)
        b_tt = _to_device(b[:-1], device)
    elif case == "nonsquare":
        a_tt = _to_device(a[:, :, :31], device)
    elif case == "key_dim":
        b_tt = _to_device(b[:, :31], device)
    elif case == "unaligned":
        b_tt = _to_device(b[:, :, :31], device)
    elif case == "state_shape":
        state_tt = _to_device(initial_state[:, :, :31], device)
    with expect_error(RuntimeError, message):
        _run(a_tt, b_tt, state_tt, groups_per_head)


def test_affine_exclusive_scan_enforces_device_worker_capacity(
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    grid = device.compute_with_storage_grid_size()
    worker_limit = grid.x * grid.y

    a, b, initial_state = _host_inputs(1, worker_limit, 32, 32)
    expected = _oracle(a, b, initial_state, 1, worker_limit)
    actual = _run(
        _to_device(a, device),
        _to_device(b, device),
        _to_device(initial_state, device),
        worker_limit,
    )
    assert_accurate(expected, ttnn.to_torch(actual), name="full-grid exclusive scan", pcc_threshold=0.999)

    group_workers = worker_limit + 1
    a, b, initial_state = _host_inputs(1, group_workers, 32, 32)
    with expect_error(RuntimeError, f"supports at most {worker_limit} group workers on this device"):
        _run(
            _to_device(a, device),
            _to_device(b, device),
            _to_device(initial_state, device),
            group_workers,
        )


@pytest.mark.parametrize("input_name", ["a", "b"])
@pytest.mark.parametrize(
    "memory_layout",
    [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
    ids=["width_sharded", "block_sharded"],
)
def test_affine_exclusive_scan_rejects_unsupported_input_sharding(
    device: ttnn.Device,
    expect_error: Callable,
    input_name: str,
    memory_layout: ttnn.TensorMemoryLayout,
) -> None:
    a, b, initial_state = _host_inputs(1, 4, 32, 32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [128, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    unsupported = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)
    a_memory = unsupported if input_name == "a" else ttnn.DRAM_MEMORY_CONFIG
    b_memory = unsupported if input_name == "b" else ttnn.DRAM_MEMORY_CONFIG

    with expect_error(RuntimeError, f"{input_name} must use interleaved or height-sharded memory"):
        _run(
            _to_device(a, device, memory_config=a_memory),
            _to_device(b, device, memory_config=b_memory),
            _to_device(initial_state, device),
            4,
        )


def test_affine_exclusive_scan_rejects_invalid_configuration(
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    a, b, initial_state = _host_inputs(1, 4, 32, 32)
    a_tt, b_tt, state_tt = _to_device(a, device), _to_device(b, device), _to_device(initial_state, device)
    with expect_error(RuntimeError, "groups_per_head must be positive"):
        _run(a_tt, b_tt, state_tt, 0)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [128, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED"):
        _run(a_tt, b_tt, state_tt, 4, memory_config=sharded)
