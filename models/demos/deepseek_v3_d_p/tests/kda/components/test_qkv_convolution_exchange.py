# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Galaxy coverage of qkv_causal_conv1d_silu_with_carry's fabric history exchange.

Every case compares Q, K, V and the replacement carry of every rank with a convolution of the whole chronological
sequence on the host, so a segment that received the wrong predecessor rows, or a rank that received the wrong
carry, fails. The cases share one persistent exchange buffer and semaphores, in the order the model would call them,
and every case also runs back to back and replays a trace, so the entry barrier and the semaphore rearm are exercised
across calls.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from models.demos.deepseek_v3_d_p.tt.kda.kda import _HistoryExchange
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    make_actual_start,
)

pytestmark = [run_for_blackhole()]

_LOCAL_ROWS = 640
_CHANNEL_CHUNK_SIZE = 64
_HISTORY_ROWS = 3
# The production input projection holds Q, K and V at column 0, followed by the decay-rank, output-gate and beta
# columns the convolution must not read.
_TRAILING_COLUMNS = 32
# Wider than one fabric packet per exchanged row, so every row goes out in more than one chunk.
_WIDTHS = (1024, 1024, 1024)
_PRODUCTION_WIDTHS = (3072, 3072, 3072)
_PRODUCTION_TRAILING_COLUMNS = 128 + 3072 + 32
_NRMSE_THRESHOLD = 0.008

_MESHES = [
    pytest.param(
        (8, 4),
        0,
        torus_xy_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="SP8xTP4-ring",
    ),
    pytest.param(
        (8, 4),
        1,
        torus_xy_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="SP4xTP8-axis1-ring",
    ),
    pytest.param(
        (8, 4),
        0,
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="SP8xTP4-line",
    ),
]


@dataclass(frozen=True)
class _Interval:
    start: int
    # Exclusive end; None is the full capacity.
    end: int | None


def _intervals(sp_size: int) -> list[_Interval]:
    capacity = sp_size * _LOCAL_ROWS
    mid_device = 4 % sp_size * _LOCAL_ROWS + _LOCAL_ROWS // 2
    last_rank = (sp_size - 1) * _LOCAL_ROWS + 32
    return [
        # Rank-aligned: no split segment.
        _Interval(0, None),
        # Production's Galaxy perf gate: a full chunk starting mid-device, so its first rank also holds the tail.
        _Interval(mid_device, mid_device + capacity),
        # Bounded intervals ending mid-device and on the head of the split rank; the carry comes from the segment
        # that holds the logical end.
        _Interval(mid_device, mid_device + capacity // 2 + 64),
        _Interval(last_rank, last_rank + capacity - _LOCAL_ROWS + 96),
        # Fewer valid rows on the final segment than the carry holds: the carry continues the previous segment, or
        # with fewer valid rows in the whole interval, the caller's history.
        _Interval(_LOCAL_ROWS, 2 * _LOCAL_ROWS + 2),
        _Interval(_LOCAL_ROWS, _LOCAL_ROWS + 2),
        # Every row of the call is valid but none is on the first rank's tail.
        _Interval(last_rank, None),
    ]


def _chronological_rows(interval: _Interval, sp_size: int) -> list[tuple[int, int]]:
    """(rank, local row) of each valid row, in chronological order."""
    topology = chronological_topology(interval.start, sp_size, _LOCAL_ROWS)
    first = topology.first_rank
    order = [(first, row) for row in range(topology.head_rows)]
    for step in range(1, sp_size):
        order.extend(((first + step) % sp_size, row) for row in range(_LOCAL_ROWS))
    order.extend((first, row) for row in range(topology.head_rows, _LOCAL_ROWS))
    valid = len(order) if interval.end is None else interval.end - interval.start
    return order[:valid]


def _reference(
    segments: torch.Tensor, history: torch.Tensor, taps: tuple[torch.Tensor, ...], order: list[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The convolution of the chronological sequence, and the three rows ending it."""
    ranks = torch.tensor([rank for rank, _ in order])
    rows = torch.tensor([row for _, row in order])
    sequence = segments[ranks, rows].unsqueeze(0).float()
    window = torch.cat((history.float(), sequence), dim=1)
    convolved = sum(window[:, tap : tap + sequence.shape[1]] * taps[tap].float() for tap in range(4))
    return F.silu(convolved)[0], window[:, -_HISTORY_ROWS:].bfloat16()


def _coordinate(sp_rank: int, other_rank: int, sp_axis: int) -> tuple[int, int]:
    return (sp_rank, other_rank) if sp_axis == 0 else (other_rank, sp_rank)


class _Exchange:
    """Device inputs, persistent exchange state and the host reference for one mesh."""

    def __init__(self, mesh_device: ttnn.MeshDevice, sp_axis: int, widths: tuple[int, int, int], trailing: int):
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.widths = widths
        mesh_shape = tuple(mesh_device.shape)
        self.sp_size = mesh_shape[sp_axis]
        self.other_size = mesh_shape[1 - sp_axis]
        channels = sum(widths)
        generator = torch.Generator().manual_seed(613)
        # [rank, row, channel]; replicated across the other mesh axis, whose ranks form independent SP rings.
        self.segments = torch.randn(self.sp_size, _LOCAL_ROWS, channels, generator=generator, dtype=torch.bfloat16)
        self.history = torch.randn(1, _HISTORY_ROWS, channels, generator=generator, dtype=torch.bfloat16)
        self.taps = tuple(torch.randn(1, 1, channels, generator=generator, dtype=torch.bfloat16) for _ in range(4))
        projected = torch.cat(
            (
                self.segments.reshape(1, self.sp_size * _LOCAL_ROWS, channels),
                torch.randn(1, self.sp_size * _LOCAL_ROWS, trailing, generator=generator, dtype=torch.bfloat16),
            ),
            dim=-1,
        )
        dims = [None, None]
        dims[sp_axis] = 1
        self.projected = ttnn.from_torch(
            projected,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=mesh_shape),
        )
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self.history_tt = ttnn.from_torch(
            self.history, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=replicate
        )
        self.taps_tt = tuple(
            ttnn.from_torch(
                tap, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate
            )
            for tap in self.taps
        )
        self.exchange = _HistoryExchange.allocate(mesh_device, channels)

    def run(self, actual_start: ttnn.Tensor, actual_end: ttnn.Tensor | None) -> tuple[ttnn.Tensor, ...]:
        return ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
            self.projected,
            self.history_tt,
            *self.taps_tt,
            *self.widths,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_CHANNEL_CHUNK_SIZE),
            actual_start=actual_start,
            exchange_buffer=self.exchange.buffer,
            exchange_semaphores=self.exchange.semaphores,
            actual_end=actual_end,
            sequence_parallel_axis=self.sp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def to_host(self, outputs: tuple[ttnn.Tensor, ...]) -> list[list[torch.Tensor]]:
        """Per device, in mesh order: the concatenated Q, K and V, and the carry."""
        q, k, v, carry = (ttnn.get_device_tensors(output) for output in outputs)
        return [
            [torch.cat([ttnn.to_torch(part) for part in parts], dim=-1)[0], ttnn.to_torch(final)]
            for *parts, final in zip(q, k, v, carry, strict=True)
        ]

    def check(self, interval: _Interval, host: list[list[torch.Tensor]]) -> None:
        order = _chronological_rows(interval, self.sp_size)
        expected, expected_carry = _reference(self.segments, self.history, self.taps, order)
        # The first rows of the sequence and of every segment read rows another rank (or the caller) supplied.
        boundaries = [
            index
            for index in range(len(order))
            if any(index - back < 0 or order[index - back][0] != order[index][0] for back in range(1, 4))
        ]
        columns = tuple(self.mesh_device.shape)[1]
        for other_rank in range(self.other_size):
            actual = torch.empty_like(expected)
            for sp_rank in range(self.sp_size):
                row, column = _coordinate(sp_rank, other_rank, self.sp_axis)
                output, carry = host[row * columns + column]
                assert_bit_identical(
                    expected_carry, carry, name=f"{interval} carry on rank {sp_rank} of ring {other_rank}"
                )
                for index, (rank, local_row) in enumerate(order):
                    if rank == sp_rank:
                        actual[index] = output[local_row].float()
            name = f"{interval} ring {other_rank}"
            assert_accurate(
                expected, actual, name=f"{name} Q, K and V", pcc_threshold=0.999, nrmse_threshold=_NRMSE_THRESHOLD
            )
            assert_accurate(
                expected[boundaries],
                actual[boundaries],
                name=f"{name} segment-start rows",
                pcc_threshold=0.999,
                nrmse_threshold=_NRMSE_THRESHOLD,
            )

    def deallocate(self) -> None:
        for tensor in (self.projected, self.history_tt, *self.taps_tt):
            ttnn.deallocate(tensor)


def _release(outputs: tuple[ttnn.Tensor, ...]) -> None:
    for tensor in outputs:
        ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device,sp_axis,device_params", _MESHES, indirect=["mesh_device", "device_params"])
def test_qkv_convolution_exchange_matches_single_sequence(
    mesh_device: ttnn.MeshDevice, sp_axis: int, device_params: dict
) -> None:
    """Accuracy against the single-sequence reference; determinism over back-to-back calls and trace replay."""
    exchange = _Exchange(mesh_device, sp_axis, _WIDTHS, _TRAILING_COLUMNS)
    try:
        for interval in _intervals(exchange.sp_size):
            actual_start = make_actual_start(mesh_device, interval.start)
            actual_end = None if interval.end is None else make_actual_start(mesh_device, interval.end)
            outputs = exchange.run(actual_start, actual_end)
            first = exchange.to_host(outputs)
            _release(outputs)
            exchange.check(interval, first)

            def assert_repeats(host: list[list[torch.Tensor]], label: str) -> None:
                for device, (reference, candidate) in enumerate(zip(first, host, strict=True)):
                    assert_bit_identical(
                        reference[0], candidate[0], name=f"{interval} {label} Q, K, V, device {device}"
                    )
                    assert_bit_identical(reference[1], candidate[1], name=f"{interval} {label} carry, device {device}")

            # Back to back, without a host round trip between the calls.
            repeated = [exchange.run(actual_start, actual_end) for _ in range(3)]
            for outputs in repeated:
                assert_repeats(exchange.to_host(outputs), "back-to-back")
                _release(outputs)

            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs = exchange.run(actual_start, actual_end)
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:
                for _ in range(2):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    assert_repeats(exchange.to_host(outputs), "trace replay")
            finally:
                ttnn.release_trace(mesh_device, trace)
                _release(outputs)
            for tensor in (actual_start, actual_end):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            logger.info(f"{interval}: single-sequence accuracy, back-to-back and trace replay PASS")
    finally:
        exchange.deallocate()


# Traced wall time of one call at Kimi-K3's per-device geometry on Galaxy SP8xTP4, from the median of the samples.
_PERF_CALLS_PER_TRACE = 20
_PERF_SAMPLES = 7
_PERF_REFERENCE_US = 69.7
_PERF_MARGIN = 0.05


@pytest.mark.parametrize("mesh_device,sp_axis,device_params", _MESHES[:1], indirect=["mesh_device", "device_params"])
def test_qkv_convolution_exchange_kimi_k3_perf(mesh_device: ttnn.MeshDevice, sp_axis: int, device_params: dict) -> None:
    exchange = _Exchange(mesh_device, sp_axis, _PRODUCTION_WIDTHS, _PRODUCTION_TRAILING_COLUMNS)
    capacity = exchange.sp_size * _LOCAL_ROWS
    start = 4 * _LOCAL_ROWS + _LOCAL_ROWS // 2
    interval = _Interval(start, start + capacity)
    actual_start = make_actual_start(mesh_device, interval.start)
    actual_end = make_actual_start(mesh_device, interval.end)
    try:
        outputs = exchange.run(actual_start, actual_end)
        exchange.check(interval, exchange.to_host(outputs))
        _release(outputs)

        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(_PERF_CALLS_PER_TRACE):
            # Released inside the trace, as the layer releases them after their use.
            _release(exchange.run(actual_start, actual_end))
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        try:
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            samples_us = []
            for _ in range(_PERF_SAMPLES):
                begin = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                samples_us.append((time.perf_counter() - begin) * 1e6 / _PERF_CALLS_PER_TRACE)
        finally:
            ttnn.release_trace(mesh_device, trace)
        median_us = statistics.median(samples_us)
        logger.info(f"qkv_causal_conv1d_silu_with_carry SP8xTP4 Kimi-K3: {median_us:.1f} us per call, {samples_us}")
        lower, upper = _PERF_REFERENCE_US * (1 - _PERF_MARGIN), _PERF_REFERENCE_US * (1 + _PERF_MARGIN)
        assert lower <= median_us <= upper, (
            f"{median_us:.1f} us per call outside [{lower:.1f}, {upper:.1f}] "
            f"(reference {_PERF_REFERENCE_US} us, margin +/- {_PERF_MARGIN * 100:.0f}%)"
        )
    finally:
        for tensor in (actual_start, actual_end):
            ttnn.deallocate(tensor)
        exchange.deallocate()
