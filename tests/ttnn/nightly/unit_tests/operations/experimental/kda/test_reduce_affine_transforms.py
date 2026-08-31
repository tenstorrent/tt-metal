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
_PRODUCTION_PERF_EXPECTED_DURATION_NS = 46_038


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
    groups_per_head: int,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.reduce_affine_transforms(
            a,
            b,
            groups_per_head,
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
@pytest.mark.parametrize("sharded_inputs", [False, True], ids=("interleaved", "height-sharded-l1"))
def test_reduce_affine_transforms_contract_and_trace(
    device: ttnn.Device,
    summary_dtype: ttnn.DataType,
    sharded_inputs: bool,
) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 32
    a, b = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected_a, expected_b = _oracle(a, b, batch_heads, groups_per_head)
    leading = batch_heads * groups_per_head
    a_memory = (
        _height_sharded_memory_config(device, leading, key_dim, key_dim) if sharded_inputs else ttnn.DRAM_MEMORY_CONFIG
    )
    b_memory = (
        _height_sharded_memory_config(device, leading, key_dim, value_dim)
        if sharded_inputs
        else ttnn.DRAM_MEMORY_CONFIG
    )
    a_tt = _to_device(a, device, summary_dtype, memory_config=a_memory)
    b_tt = _to_device(b, device, summary_dtype, memory_config=b_memory)
    snapshots = (ttnn.to_torch(a_tt).clone(), ttnn.to_torch(b_tt).clone())

    first = _run(a_tt, b_tt, groups_per_head)
    for output, shape in zip(first, ((batch_heads, key_dim, key_dim), (batch_heads, key_dim, value_dim)), strict=True):
        assert output.dtype == ttnn.float32
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(ttnn.to_torch(output).shape) == shape
        assert output.buffer_address() not in (a_tt.buffer_address(), b_tt.buffer_address())

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = _run(a_tt, b_tt, groups_per_head)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, golden, actual_tt, traced_tt in zip(("A", "B"), (expected_a, expected_b), first, traced, strict=True):
        actual = ttnn.to_torch(actual_tt)
        assert_accurate(golden, actual, name=f"{summary_dtype} reduced {name}", pcc_threshold=0.999)
        assert_bit_identical(actual, ttnn.to_torch(traced_tt), name=f"{name} trace replay")

    assert_bit_identical(snapshots[0], ttnn.to_torch(a_tt), name="a immutability")
    assert_bit_identical(snapshots[1], ttnn.to_torch(b_tt), name="b immutability")
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    ("batch_heads", "groups_per_head", "key_dim", "value_dim", "summary_dtype", "sharded_inputs"),
    [
        pytest.param(2, 1, 32, 32, ttnn.float32, False, id="bh2-g1-k32-v32-fp32-interleaved"),
        pytest.param(2, 3, 32, 32, ttnn.bfloat16, True, id="bh2-g3-k32-v32-bf16-height-sharded-l1"),
        pytest.param(3, 2, 32, 64, ttnn.float32, False, id="bh3-g2-k32-v64-fp32-interleaved"),
        pytest.param(2, 2, 160, 32, ttnn.bfloat16, True, id="bh2-g2-k160-v32-bf16-height-sharded-l1"),
        pytest.param(2, 2, 32, 160, ttnn.float32, False, id="bh2-g2-k32-v160-fp32-interleaved"),
    ],
)
def test_reduce_affine_transforms_shape_accuracy(
    device: ttnn.Device,
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
    summary_dtype: ttnn.DataType,
    sharded_inputs: bool,
) -> None:
    a, b = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim)
    expected = _oracle(a, b, batch_heads, groups_per_head)
    leading = batch_heads * groups_per_head
    a_memory = (
        _height_sharded_memory_config(device, leading, key_dim, key_dim) if sharded_inputs else ttnn.DRAM_MEMORY_CONFIG
    )
    b_memory = (
        _height_sharded_memory_config(device, leading, key_dim, value_dim)
        if sharded_inputs
        else ttnn.DRAM_MEMORY_CONFIG
    )
    a_tt = _to_device(a, device, summary_dtype, memory_config=a_memory)
    b_tt = _to_device(b, device, summary_dtype, memory_config=b_memory)

    outputs = _run(a_tt, b_tt, groups_per_head)

    for name, golden, output in zip(("A", "B"), expected, outputs, strict=True):
        assert_accurate(
            golden,
            ttnn.to_torch(output),
            name=f"{summary_dtype} shape-sweep reduced {name}",
            pcc_threshold=0.999,
        )


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
def test_reduce_affine_transforms_is_device_deterministic(device: ttnn.Device, summary_dtype: ttnn.DataType) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1441)
    a_tt, b_tt = (_to_device(tensor, device, summary_dtype) for tensor in host)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)

    def run() -> tuple[ttnn.Tensor, ...]:
        return _run(a_tt, b_tt, case.groups_per_head)

    reference_outputs, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{summary_dtype} reduced outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("A", "B"), expected, outputs, strict=True):
        assert_accurate(golden, output, name=f"{summary_dtype} reduced {name}", pcc_threshold=0.999)
    for output in reference_outputs:
        ttnn.deallocate(output)


def test_reduce_affine_transforms_cache_hit_rebinds_fresh_tensors(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host_a = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1911)
    host_b = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=1912)
    device_a = tuple(_to_device(tensor, device) for tensor in host_a)
    device_b = tuple(_to_device(tensor, device) for tensor in host_b)

    output_a = _run(*device_a, case.groups_per_head)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(*device_b, case.groups_per_head)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(device_a, device_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(output_a, output_b, strict=True))
    expected_a = _oracle(*host_a, case.batch_heads, case.groups_per_head)
    expected_b = _oracle(*host_b, case.batch_heads, case.groups_per_head)
    for name, golden_a, golden_b, actual_a_tt, actual_b_tt in zip(
        ("A", "B"), expected_a, expected_b, output_a, output_b, strict=True
    ):
        actual_a = ttnn.to_torch(actual_a_tt)
        actual_b = ttnn.to_torch(actual_b_tt)
        assert_accurate(golden_a, actual_a, name=f"{name} cache miss tensors", pcc_threshold=0.999)
        assert_accurate(golden_b, actual_b, name=f"{name} cache hit fresh tensors", pcc_threshold=0.999)
        assert not torch.equal(actual_a, actual_b)


def test_reduce_affine_transforms_default_compute_config_matches_explicit_defaults(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=817)
    a_tt, b_tt = (_to_device(tensor, device) for tensor in host)
    implicit = _run(a_tt, b_tt, case.groups_per_head)
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
    explicit = _run(a_tt, b_tt, case.groups_per_head, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    for name, implicit_output, explicit_output in zip(("A", "B"), implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_output),
            ttnn.to_torch(explicit_output),
            name=f"{name} implicit vs explicit compute defaults",
        )


def test_reduce_affine_transforms_approximate_math_uses_distinct_accurate_program(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=818)
    a_tt, b_tt = (_to_device(tensor, device) for tensor in host)
    exact = _run(a_tt, b_tt, case.groups_per_head)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = _run(a_tt, b_tt, case.groups_per_head, compute_kernel_config=approximate_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    for name, golden, exact_output, approximate_output in zip(("A", "B"), expected, exact, approximate, strict=True):
        assert_accurate(golden, ttnn.to_torch(exact_output), name=f"{name} exact math", pcc_threshold=0.999)
        assert_accurate(golden, ttnn.to_torch(approximate_output), name=f"{name} approximate math", pcc_threshold=0.999)


def test_reduce_affine_transforms_rejects_unsupported_compute_config(
    device: ttnn.Device, expect_error: Callable
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim)
    a_tt, b_tt = (_to_device(tensor, device) for tensor in host)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(a_tt, b_tt, case.groups_per_head, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_reduce_affine_transforms_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_PERF_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for affine-transform reduction performance checks")

    host = _host_inputs(case.batch_heads, case.groups_per_head, case.key_dim, case.value_dim, seed=117)
    expected = _oracle(*host, case.batch_heads, case.groups_per_head)
    a_tt, b_tt = (_to_device(tensor, device, ttnn.bfloat16) for tensor in host)
    production_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return _run(a_tt, b_tt, case.groups_per_head, compute_kernel_config=production_compute_config)

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(tuple(output.shape) for output in outputs) == (
        (case.batch_heads, case.key_dim, case.key_dim),
        (case.batch_heads, case.key_dim, case.value_dim),
    )
    assert all(output.dtype == ttnn.float32 for output in outputs)
    for name, golden, output in zip(("A", "B"), expected, outputs, strict=True):
        assert_accurate(golden, ttnn.to_torch(output), name=f"production reduced {name}", pcc_threshold=0.999)
    logger.info(
        f"affine-transform reduction {case.case_id}: duration={duration_ns:.0f} ns, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )
    lower = _PRODUCTION_PERF_EXPECTED_DURATION_NS * (1 - _PRODUCTION_PERF_MARGIN)
    upper = _PRODUCTION_PERF_EXPECTED_DURATION_NS * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {_PRODUCTION_PERF_EXPECTED_DURATION_NS} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


def test_reduce_affine_transforms_matches_composed_ttnn_baseline(device: ttnn.Device) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 64
    a, b = _host_inputs(batch_heads, groups_per_head, key_dim, value_dim, seed=117)
    expected = _oracle(a, b, batch_heads, groups_per_head)
    fused = _run(_to_device(a, device), _to_device(b, device), groups_per_head)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        composed = _composed_ttnn_baseline(a, b, device, batch_heads, groups_per_head)
    ttnn.synchronize_device(device)
    for name, golden, fused_tt, composed_tt in zip(("A", "B"), expected, fused, composed, strict=True):
        assert_accurate(golden, ttnn.to_torch(fused_tt), name=f"fused {name}", pcc_threshold=0.999)
        assert_accurate(golden, ttnn.to_torch(composed_tt), name=f"composed TTNN {name}", pcc_threshold=0.999)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_a", "a must be an allocated device tensor"),
        ("host_b", "b must be an allocated device tensor"),
        ("dtype", "inputs must have matching dtypes"),
        ("layout", "b must use TILE layout"),
        ("rank", "inputs must be rank 3"),
        ("leading", "matching leading dimensions"),
        ("nondivisible", "leading dimension must be divisible"),
        ("nonsquare", "a must contain square"),
        ("key_dim", "matching K dimensions"),
        ("unaligned", "K and V must be positive and tile aligned"),
    ],
)
def test_reduce_affine_transforms_rejects_invalid_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    a, b = _host_inputs(1, 4, 32, 32)
    groups_per_head = 4
    a_tt = _to_device(a, device)
    b_tt = _to_device(b, device)

    if case == "host_a":
        a_tt = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "host_b":
        b_tt = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "dtype":
        b_tt = _to_device(b, device, ttnn.bfloat16)
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
    with expect_error(RuntimeError, message):
        _run(a_tt, b_tt, groups_per_head)


def test_reduce_affine_transforms_enforces_device_worker_capacity(
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    grid = device.compute_with_storage_grid_size()
    worker_limit = grid.x * grid.y

    a, b = _host_inputs(1, worker_limit, 32, 32)
    expected_a, expected_b = _oracle(a, b, 1, worker_limit)
    actual_a, actual_b = _run(_to_device(a, device), _to_device(b, device), worker_limit)
    assert_accurate(expected_a, ttnn.to_torch(actual_a), name="full-grid reduced A", pcc_threshold=0.999)
    assert_accurate(expected_b, ttnn.to_torch(actual_b), name="full-grid reduced B", pcc_threshold=0.999)

    group_workers = worker_limit + 1
    a, b = _host_inputs(1, group_workers, 32, 32)
    with expect_error(RuntimeError, f"supports at most {worker_limit} group workers on this device"):
        _run(_to_device(a, device), _to_device(b, device), group_workers)


@pytest.mark.parametrize("input_name", ["a", "b"])
@pytest.mark.parametrize(
    "memory_layout",
    [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
    ids=["width_sharded", "block_sharded"],
)
def test_reduce_affine_transforms_rejects_unsupported_input_sharding(
    device: ttnn.Device,
    expect_error: Callable,
    input_name: str,
    memory_layout: ttnn.TensorMemoryLayout,
) -> None:
    a, b = _host_inputs(1, 4, 32, 32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [128, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    unsupported = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)
    a_memory = unsupported if input_name == "a" else ttnn.DRAM_MEMORY_CONFIG
    b_memory = unsupported if input_name == "b" else ttnn.DRAM_MEMORY_CONFIG
    a_tt = _to_device(a, device, memory_config=a_memory)
    b_tt = _to_device(b, device, memory_config=b_memory)

    with expect_error(RuntimeError, f"{input_name} must use interleaved or height-sharded memory"):
        _run(a_tt, b_tt, 4)


def test_reduce_affine_transforms_rejects_invalid_configuration(
    device: ttnn.Device,
    expect_error: Callable,
) -> None:
    a, b = _host_inputs(1, 4, 32, 32)
    a_tt, b_tt = _to_device(a, device), _to_device(b, device)
    with expect_error(RuntimeError, "groups_per_head must be positive"):
        _run(a_tt, b_tt, 0)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [32, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"):
        _run(a_tt, b_tt, 4, memory_config=sharded)
