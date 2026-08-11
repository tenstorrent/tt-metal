# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for the experimental KDA affine exclusive scan."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]


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
) -> ttnn.Tensor:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.affine_exclusive_scan(
            a,
            b,
            initial_state,
            groups_per_head,
            memory_config=memory_config,
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
@pytest.mark.parametrize(
    ("batch_heads", "groups_per_head", "key_dim", "value_dim"),
    [(2, 4, 32, 32), (3, 2, 32, 64)],
)
def test_affine_exclusive_scan_contract_cache_trace_and_determinism(
    device: ttnn.Device,
    summary_dtype: ttnn.DataType,
    sharded_inputs: bool,
    batch_heads: int,
    groups_per_head: int,
    key_dim: int,
    value_dim: int,
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

    cache_entries = device.num_program_cache_entries()
    repeated = _run(a_tt, b_tt, state_tt, groups_per_head, memory_config=output_memory)
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries

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
    assert_bit_identical(actual, ttnn.to_torch(repeated), name="eager repeat")
    assert_bit_identical(actual, ttnn.to_torch(traced), name="trace replay")
    for name, snapshot, tensor in zip(("a", "b", "initial_state"), snapshots, (a_tt, b_tt, state_tt), strict=True):
        assert_bit_identical(snapshot, ttnn.to_torch(tensor), name=f"{name} immutability")
    ttnn.release_trace(device, trace_id)


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
    with expect_error(RuntimeError, "output memory configuration must be interleaved"):
        _run(a_tt, b_tt, state_tt, 4, memory_config=sharded)
