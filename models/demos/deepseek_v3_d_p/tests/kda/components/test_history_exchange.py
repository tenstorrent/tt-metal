# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sequence-parallel convolution, with its fabric history exchange, matches a single-sequence reference.

Rings (torus) and lines (unwrapped 2D and 1D fabric) take different routes. Back-to-back calls reuse the persistent
exchange buffer and semaphores, so a rank running ahead must not disturb a slower peer, and every chronology case
(split segments, short final segments, one token) must convolve and carry the right rows.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    fabric_1d_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import _HistoryExchange
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    make_actual_start,
    qkv_device_inputs,
    qkv_reference,
)

LOCAL_ROWS = 64
# Per call, in call order, from the capacity C: (actual_start, interval length, None for the full capacity).
INTERVALS = [
    lambda C: (0, None),
    lambda C: (32, None),
    lambda C: (LOCAL_ROWS, None),
    # One valid row on the second rank: the carry continues the first rank's rows.
    lambda C: (0, LOCAL_ROWS + 1),
    # Two valid rows in a split tail.
    lambda C: (32, C - 30),
    lambda C: (96, 1),
    lambda C: (0, None),
]


def _shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _global_positions(rank: int, start: int, sp_size: int) -> torch.Tensor:
    """Chronological position of each local row on `rank`: the first rank holds the head and a wrapped tail."""
    first_rank = start // LOCAL_ROWS % sp_size
    head_rows = LOCAL_ROWS - start % LOCAL_ROWS
    step = (rank - first_rank) % sp_size
    rows = torch.arange(LOCAL_ROWS)
    if step == 0:
        return torch.where(rows < head_rows, rows, rows + (sp_size - 1) * LOCAL_ROWS)
    return head_rows + (step - 1) * LOCAL_ROWS + rows


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="8x4-torus-xy",
        ),
        # Without wraparound links each SP axis is a line: the last rank reaches rank 0 backward.
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="8x4-fabric-2d-line",
        ),
        pytest.param(
            (8, 4),
            fabric_1d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="8x4-fabric-1d-line",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("axis", [0, 1])
# Rows of 256 bytes fit one fabric packet; rows of 6 KiB are split across packets.
@pytest.mark.parametrize("widths", [(32, 32, 64), (1024, 1024, 1024)], ids=["narrow", "wide"])
def test_history_exchange_matches_reference(mesh_device, axis, widths):
    sp_size = tuple(mesh_device.shape)[axis]
    columns = tuple(mesh_device.shape)[1]
    exchange = _HistoryExchange.allocate(mesh_device, sum(widths))
    program_config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=32)

    # Every rank holds the same local rows, so the chronological sequence repeats them in the rotated rank order.
    # No synchronization or collective between calls: a rank may run a call ahead of its peers. Each interval runs
    # twice in a row.
    calls = []
    for call, interval in enumerate(INTERVALS):
        start, length = interval(sp_size * LOCAL_ROWS)
        (rows, history, taps), (projection, history_tt, taps_tt) = qkv_device_inputs(
            mesh_device, widths=widths, sequence=LOCAL_ROWS, history_rows=3, seed=4099 + call
        )
        projection = ttnn.to_layout(projection, ttnn.TILE_LAYOUT)
        actual_start = make_actual_start(mesh_device, start)
        actual_end = None if length is None else make_actual_start(mesh_device, start + length)
        for repeat in range(2):
            outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
                projection,
                history_tt,
                *taps_tt,
                *widths,
                program_config=program_config,
                actual_start=actual_start,
                exchange_buffer=exchange.buffer,
                exchange_semaphores=exchange.semaphores,
                actual_end=actual_end,
                sequence_parallel_axis=axis,
            )
            calls.append((start, length, repeat, rows, history, taps, outputs))
    ttnn.synchronize_device(mesh_device)

    for start, length, repeat, rows, history, taps, outputs in calls:
        valid = sp_size * LOCAL_ROWS if length is None else length
        order = torch.cat([_global_positions(rank, start, sp_size) for rank in range(sp_size)])
        sequence = torch.empty(1, sp_size * LOCAL_ROWS, rows.shape[-1], dtype=rows.dtype)
        sequence[:, order] = rows.repeat(1, sp_size, 1)
        expected = qkv_reference(sequence[:, :valid], history, taps, widths)
        expected_carry = torch.cat((history, sequence[:, :valid]), dim=1)[:, -3:]
        label = f"start={start} length={length} repeat={repeat} sp={sp_size}"
        for name, golden, output in zip(("q", "k", "v"), expected, outputs[:3], strict=True):
            for shard, result in enumerate(_shards(output)):
                rank = shard // columns if axis == 0 else shard % columns
                positions = _global_positions(rank, start, sp_size)
                live = positions < valid
                if live.any():
                    assert_accurate(golden[:, positions[live]], result[:, live], name=f"{name} {label} shard={shard}")
        for shard, carry in enumerate(_shards(outputs[3])):
            assert_bit_identical(expected_carry, carry, name=f"carry {label} shard={shard}")
