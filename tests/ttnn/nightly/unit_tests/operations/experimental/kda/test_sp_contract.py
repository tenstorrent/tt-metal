# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared KDA APIs derive SP resets from actual_start on real mesh coordinates.

Replaces the manual indicator/wrap tests. Every physical rank and TP replica is
checked, including live compact-summary slots and group-aligned tail reseeding.
Keep mesh tests separate from the ordinary tests' module-scoped device fixture.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    host_protocol,
    initial_state,
    recurrent_oracle,
    summary_oracle,
    to_device,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    height_sharded_memory_config,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    qkv_device_inputs,
    qkv_reference,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    segmented_summary_oracle,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_reset_replaces_large_carry_exactly(mesh_device):
    """A unit seed must survive replacement of a 2**26 carry without cancellation."""
    host = list(host_protocol(1, 2, 32, 32))
    for tensor in host:
        tensor.zero_()
    host[2][:] = torch.eye(32)
    host[5].fill_(1)
    host[6][:] = torch.eye(32)
    inputs = device_protocol(host, mesh_device)
    seed = to_device(torch.full((1, 32, 32), float(2**26)), mesh_device)
    tail = to_device(torch.ones(1, 32, 32), mesh_device)
    output, final_state = ttnn.experimental.kda.recurrent_chunk_scan(
        *inputs, seed, tail_entry_states=tail, actual_start=make_actual_start(mesh_device, 32), sequence_parallel_axis=0
    )
    for index, (y, state) in enumerate(zip(_shards(output), _shards(final_state), strict=True)):
        expected = 1.0 if _rank(index, mesh_device, 0) == 0 else float(2**26)
        assert torch.equal(state, torch.full_like(state, expected))
        assert torch.equal(y[:, 1], torch.full_like(y[:, 1], expected))


def _shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(tensor)]


def _rank(index: int, mesh_device: ttnn.MeshDevice, axis: int) -> int:
    return index // tuple(mesh_device.shape)[1] if axis == 0 else index % tuple(mesh_device.shape)[1]


def _assert_immutable(tensors: list[ttnn.Tensor], before: list[list[torch.Tensor]]) -> None:
    for tensor, snapshots in zip(tensors, before, strict=True):
        for expected, actual in zip(snapshots, _shards(tensor), strict=True):
            assert_bit_identical(expected, actual, name="immutable SP input")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("groups,dim", [(1, 32), (4, 32), (4, 128)])
def test_sp_summary_prefix_and_scan(mesh_device: ttnn.MeshDevice, axis: int, groups: int, dim: int) -> None:
    """Noncommuting summaries compose correctly, including unwritten inactive slots.

    Independently scan each host chunk, resetting only at the specified token.
    Fresh allocations change both payload and actual_start without growing cache.
    """
    heads, chunks = 2, 4
    local_rows = groups * chunks * 32
    sp_size = tuple(mesh_device.shape)[axis]
    positions = [
        rank * local_rows + shift
        for rank in range(sp_size)
        for shift in sorted({0, 32, local_rows // 2, local_rows - 32})
    ]
    previous = []
    cache_entries = None
    for generation, actual_start_value in enumerate(positions):
        host = host_protocol(heads * groups, chunks, dim, dim, seed=4301 + generation)
        head_seed = initial_state(heads, dim, dim, seed=5301 + generation)
        tail_seed = 20 * initial_state(heads, dim, dim, seed=6301 + generation)
        inputs = list(device_protocol(host, mesh_device))
        head_tt, tail_tt = to_device(head_seed, mesh_device), to_device(tail_seed, mesh_device)
        actual_start = make_actual_start(mesh_device, actual_start_value)
        owned = [*inputs, head_tt, tail_tt, actual_start]
        before = [_shards(t) for t in owned]
        memory = (
            height_sharded_memory_config(mesh_device, heads * groups, dim, dim)
            if dim == 128
            else ttnn.DRAM_MEMORY_CONFIG
        )
        summaries = ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs,
            groups_per_head=groups,
            actual_start=actual_start,
            sequence_parallel_axis=axis,
            memory_config=memory,
        )
        assert len(summaries) == 4
        assert all(t.dtype == ttnn.bfloat16 and t.memory_config() == memory for t in summaries)
        entries = ttnn.experimental.kda.affine_exclusive_scan(
            *summaries[:2],
            head_tt,
            groups,
            tail_a=summaries[2],
            tail_b=summaries[3],
            tail_entry_states=tail_tt,
            actual_start=actual_start,
            sequence_parallel_axis=axis,
            local_rows=local_rows,
        )
        output, final = ttnn.experimental.kda.recurrent_chunk_scan(
            *inputs,
            entries,
            groups_per_head=groups,
            tail_entry_states=tail_tt,
            actual_start=actual_start,
            sequence_parallel_axis=axis,
        )
        ttnn.synchronize_device(mesh_device)
        if cache_entries is None:
            cache_entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == cache_entries
        assert entries.dtype == final.dtype == ttnn.float32
        assert output.dtype == ttnn.bfloat16
        assert all(t.memory_config() == ttnn.DRAM_MEMORY_CONFIG for t in (entries, output, final))
        got_summaries = [_shards(t) for t in summaries]
        for shard, (got_entries, got_output, got_final) in enumerate(
            zip(_shards(entries), _shards(output), _shards(final), strict=True)
        ):
            rank = _rank(shard, mesh_device, axis)
            first_rank = actual_start_value // local_rows % sp_size
            split = (
                (local_rows - actual_start_value % local_rows) // 32
                if rank == first_rank and actual_start_value % local_rows
                else groups * chunks
            )
            expected_parts = segmented_summary_oracle(host, groups, chunks, split)
            expected_entries, expected_output, expected_final = [], [], []
            for head in range(heads):
                state = head_seed[head : head + 1]
                for group in range(groups):
                    folded = head * groups + group
                    group_start = group * chunks
                    protocol = tuple(t[folded : folded + 1] for t in host)
                    if group_start == split:
                        state = tail_seed[head : head + 1]
                    expected_entries.append(state)
                    inside = split - group_start
                    if 0 < inside < chunks:
                        head_output, _ = recurrent_oracle(tuple(t[:, :inside] for t in protocol), state)
                        tail_output, state = recurrent_oracle(
                            tuple(t[:, inside:] for t in protocol), tail_seed[head : head + 1]
                        )
                        expected_output.append(torch.cat((head_output, tail_output), dim=1))
                    else:
                        group_output, state = recurrent_oracle(protocol, state)
                        expected_output.append(group_output)
                    expected_final.append(state)
                    for part in range(4):
                        live = group_start < split if part < 2 else (group + 1) * chunks > split
                        if live:
                            assert_accurate(
                                expected_parts[part][folded].bfloat16().float(),
                                got_summaries[part][shard][folded].float(),
                                name=f"SP summary rank={rank} group={group} part={part}",
                            )
                    if 0 < inside < chunks:
                        a, b, tail_a, tail_b = [parts[shard][folded].float() for parts in got_summaries]
                        full_a, full_b = summary_oracle(protocol)
                        assert_accurate(full_a[0], tail_a @ a, name="summary composition A")
                        assert_accurate(full_b[0], tail_a @ b + tail_b, name="summary composition B")
            for expected, actual, name in (
                (torch.cat(expected_entries), got_entries, "entries"),
                (torch.cat(expected_output), got_output, "outputs"),
                (torch.cat(expected_final), got_final, "final states"),
            ):
                assert_accurate(
                    expected,
                    actual,
                    name=f"SP {name} rank={rank} actual_start={actual_start_value}",
                    rmse_threshold=0.04,
                )
        _assert_immutable(owned, before)
        current = [*owned, *summaries, entries, output, final]
        if previous:
            for old, new in zip(previous, current, strict=True):
                assert all(
                    a.buffer_address() != b.buffer_address()
                    for a, b in zip(ttnn.get_device_tensors(old), ttnn.get_device_tensors(new), strict=True)
                ), "fresh cache bindings must use different addresses on every device"
            for tensor in previous:
                ttnn.deallocate(tensor)
        previous = current
    for tensor in previous:
        ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("key_dim,value_dim", [(32, 128), (128, 32)])
def test_sp_rectangular_recurrent_reset(mesh_device, axis, key_dim, value_dim):
    """A distant tail seed detects both missing and unintended resets."""
    host = host_protocol(2, 5, key_dim, value_dim)
    head_seed = initial_state(2, key_dim, value_dim)
    tail_seed = 20 * initial_state(2, key_dim, value_dim, seed=789)
    inputs = device_protocol(host, mesh_device)
    head_tt, tail_tt = to_device(head_seed, mesh_device), to_device(tail_seed, mesh_device)
    for first_rank in range(tuple(mesh_device.shape)[axis]):
        for split in (1, 2, 4):
            actual_start = make_actual_start(mesh_device, first_rank * 160 + 160 - split * 32)
            output, state = ttnn.experimental.kda.recurrent_chunk_scan(
                *inputs, head_tt, tail_entry_states=tail_tt, actual_start=actual_start, sequence_parallel_axis=axis
            )
            head_output, _ = recurrent_oracle(tuple(t[:, :split] for t in host), head_seed)
            tail_output, tail_final = recurrent_oracle(tuple(t[:, split:] for t in host), tail_seed)
            ordinary = recurrent_oracle(host, head_seed)
            for shard, (got_output, got_state) in enumerate(zip(_shards(output), _shards(state), strict=True)):
                expected = (
                    (torch.cat((head_output, tail_output), dim=1), tail_final)
                    if _rank(shard, mesh_device, axis) == first_rank
                    else ordinary
                )
                for golden, got in zip(expected, (got_output, got_state), strict=True):
                    assert_accurate(golden, got, name=f"rectangular reset rank={first_rank} split={split}")
            for tensor in (actual_start, output, state):
                ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("axis", [0, 1])
def test_sp_convolution_rebinds_histories(mesh_device, axis):
    widths = (32, 32, 32)
    previous = []
    cache_entries = None
    for actual_start_value in (0, 32, 64, 96, 32):
        (host, history, taps), (input_tt, history_tt, taps_tt) = qkv_device_inputs(
            mesh_device, widths=widths, sequence=64, history_rows=3, seed=2011 + actual_start_value
        )
        predecessor = history + 2
        predecessor_tt = to_device(predecessor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Match the BF16 payload used by the native operation exactly.
        predecessor = predecessor.bfloat16()
        actual_start = make_actual_start(mesh_device, actual_start_value)
        owned = [input_tt, history_tt, *taps_tt, predecessor_tt, actual_start]
        before = [_shards(t) for t in owned]
        outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            input_tt,
            history_tt,
            *taps_tt,
            *widths,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=96),
            actual_start=actual_start,
            sequence_parallel_axis=axis,
            predecessor_carry=predecessor_tt,
        )
        ttnn.synchronize_device(mesh_device)
        if cache_entries is None:
            cache_entries = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == cache_entries
        for shard, got in enumerate(zip(*[_shards(t) for t in outputs], strict=True)):
            rank = _rank(shard, mesh_device, axis)
            first_rank = actual_start_value // 64 % tuple(mesh_device.shape)[axis]
            if rank == first_rank and actual_start_value % 64:
                expected = tuple(
                    torch.cat(parts, dim=1)
                    for parts in zip(
                        qkv_reference(host[:, :32], history, taps, widths),
                        qkv_reference(host[:, 32:], predecessor, taps, widths),
                        strict=True,
                    )
                )
            else:
                expected = qkv_reference(host, history if rank == first_rank else predecessor, taps, widths)
            for golden, actual in zip(expected, got, strict=True):
                assert_accurate(golden, actual, name=f"SP convolution rank={rank}")
        _assert_immutable(owned, before)
        current = [*owned, *outputs]
        if previous:
            for old, new in zip(previous, current, strict=True):
                assert all(
                    a.buffer_address() != b.buffer_address()
                    for a, b in zip(ttnn.get_device_tensors(old), ttnn.get_device_tensors(new), strict=True)
                ), "fresh cache bindings must use different addresses on every device"
            for tensor in previous:
                ttnn.deallocate(tensor)
        previous = current
    for tensor in previous:
        ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_sp_payload_contracts(mesh_device, expect_error):
    """Missing and stray SP payloads fail at the owning operation boundary."""
    inputs = device_protocol(host_protocol(2, 4, 32, 32), mesh_device)
    seed = to_device(initial_state(2, 32, 32), mesh_device)
    actual_start = make_actual_start(mesh_device, 32)
    for kwargs in ({"actual_start": actual_start}, {"tail_entry_states": seed}):
        with expect_error(TypeError, "incompatible function arguments"):
            ttnn.experimental.kda.recurrent_chunk_scan(*inputs, seed, **kwargs)
    a, b, tail_a, tail_b = ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, actual_start=actual_start)
    assert all(t.dtype == ttnn.bfloat16 for t in (a, b, tail_a, tail_b))
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, actual_start=None)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.reduce_affine_transforms(a, b, 1, actual_start=None, local_rows=128)
    for missing in ("actual_start", "tail_a", "tail_b", "tail_entry_states"):
        kwargs = dict(actual_start=actual_start, tail_a=a, tail_b=b, tail_entry_states=seed)
        del kwargs[missing]
        with expect_error(TypeError, "incompatible function arguments"):
            ttnn.experimental.kda.affine_exclusive_scan(a, b, seed, 1, local_rows=128, **kwargs)
    for local_rows in (0, 31, 32):
        with expect_error(RuntimeError, "local_rows"):
            ttnn.experimental.kda.affine_exclusive_scan(
                a,
                b,
                to_device(initial_state(1, 32, 32), mesh_device),
                2,
                actual_start=actual_start,
                tail_a=a,
                tail_b=b,
                tail_entry_states=to_device(initial_state(1, 32, 32), mesh_device),
                local_rows=local_rows,
            )
    _, (input_tt, history, taps) = qkv_device_inputs(mesh_device, widths=(32, 32, 32), sequence=64, history_rows=3)
    for kwargs in ({"actual_start": actual_start}, {"predecessor_carry": history}):
        with expect_error(TypeError, "incompatible function arguments"):
            ttnn.experimental.kda.qkv_causal_conv1d_silu(
                input_tt,
                history,
                *taps,
                32,
                32,
                32,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=96),
                **kwargs,
            )
    wrong_history = to_device(torch.zeros(1, 6, 96), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(RuntimeError, "history must be"):
        ttnn.experimental.kda.qkv_causal_conv1d_silu(
            input_tt,
            wrong_history,
            *taps,
            32,
            32,
            32,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=96),
            actual_start=actual_start,
            predecessor_carry=wrong_history,
        )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_sp_affine_rectangular_live_slots(mesh_device, axis, dtype):
    """Both supported summary dtypes ignore poisoned inactive head/tail slots."""
    heads, groups, key_dim, value_dim = 2, 4, 32, 64
    local_rows = groups * 4 * 32
    sp_size = tuple(mesh_device.shape)[axis]
    generator = torch.Generator().manual_seed(9134)
    host_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    a = (0.8 * torch.eye(key_dim) + 0.02 * torch.randn(heads * groups, key_dim, key_dim, generator=generator)).to(
        host_dtype
    )
    b = (0.03 * torch.randn(heads * groups, key_dim, value_dim, generator=generator)).to(host_dtype)
    tail_a = (0.7 * torch.eye(key_dim) + 0.03 * torch.randn(heads * groups, key_dim, key_dim, generator=generator)).to(
        host_dtype
    )
    tail_b = (0.04 * torch.randn(heads * groups, key_dim, value_dim, generator=generator)).to(host_dtype)
    head_seed = initial_state(heads, key_dim, value_dim)
    tail_seed = 20 * initial_state(heads, key_dim, value_dim, seed=901)
    # Identical physical payloads on each SP rank cannot model inactive slots:
    # construct rank-specific tensors and replicate them across the TP axis.
    for first_rank in range(sp_size):
        for split_chunk in (4, 9, 12):
            split_group, inside = divmod(split_chunk, 4)
            reset_group = split_group if inside else split_group - 1
            ranks = [[t.clone() for t in (a, b, tail_a, tail_b)] for _ in range(sp_size)]
            expected = []
            for rank in range(sp_size):
                entries = []
                for head in range(heads):
                    state = head_seed[head]
                    for group in range(groups):
                        folded = head * groups + group
                        entries.append(state)
                        if rank == first_rank and group == reset_group:
                            state = (
                                tail_seed[head]
                                if not inside
                                else tail_a[folded].float() @ tail_seed[head] + tail_b[folded].float()
                            )
                        elif rank == first_rank and group > reset_group:
                            state = tail_a[folded].float() @ state + tail_b[folded].float()
                        else:
                            state = a[folded].float() @ state + b[folded].float()
                        if rank == first_rank and group >= split_group + bool(inside):
                            ranks[rank][0][folded] = ranks[rank][1][folded] = torch.nan
                        if rank != first_rank or group < split_group:
                            ranks[rank][2][folded] = ranks[rank][3][folded] = torch.nan
                expected.append(torch.stack(entries))
            dims = [None, None]
            dims[axis] = 0
            tensors = [
                ttnn.from_torch(
                    torch.cat([rank[part] for rank in ranks]),
                    device=mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims)
                    ),
                )
                for part in range(4)
            ]
            head_tt, tail_tt = to_device(head_seed, mesh_device), to_device(tail_seed, mesh_device)
            actual_start = make_actual_start(mesh_device, first_rank * local_rows + local_rows - 32 * split_chunk)
            result = ttnn.experimental.kda.affine_exclusive_scan(
                *tensors[:2],
                head_tt,
                groups,
                tail_a=tensors[2],
                tail_b=tensors[3],
                tail_entry_states=tail_tt,
                actual_start=actual_start,
                sequence_parallel_axis=axis,
                local_rows=local_rows,
            )
            for shard, actual in enumerate(_shards(result)):
                assert_accurate(
                    expected[_rank(shard, mesh_device, axis)],
                    actual,
                    name=f"affine live slots {dtype} rank={first_rank} split={split_chunk}",
                )
            for tensor in (*tensors, head_tt, tail_tt, actual_start, result):
                ttnn.deallocate(tensor)
