# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoudBox and QuietBox coverage of qkv_causal_conv1d_silu_with_carry's fabric history exchange.

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
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from models.demos.deepseek_v3_d_p.tt.kda.kda import _HistoryExchange
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    make_actual_start,
)

pytestmark = [run_for_blackhole()]

_CHANNEL_CHUNK_SIZE = 64
_HISTORY_ROWS = 3
_NRMSE_THRESHOLD = 0.008


@dataclass(frozen=True)
class _Geometry:
    """Rows and channels of one device's call, and where its input projection and Q, K and V live."""

    local_rows: int
    widths: tuple[int, int, int]
    # The input projection holds Q, K and V at column 0, followed by the decay-rank, output-gate and beta columns
    # the convolution must not read.
    trailing_columns: int
    input_memory_config: ttnn.MemoryConfig
    output_memory_config: ttnn.MemoryConfig


def _kimi_k3(local_rows: int, tensor_parallel: int, memory_config: ttnn.MemoryConfig) -> _Geometry:
    """Kimi-K3's per-device call: 96 heads of 128 split over the TP ranks; the decay rank is 128."""
    heads = 96 // tensor_parallel
    width = heads * 128
    beta_columns = -(-heads // 32) * 32
    return _Geometry(local_rows, (width, width, width), 128 + width + beta_columns, memory_config, memory_config)


# LoudBox's production layout, SP2xTP4. Q, K and V exceed KDA_QKV_L1_BYTES_PER_CORE and the projection is not the
# tuned L1 one at 2560 rows, so both stay in DRAM, as in the layer.
_LOUDBOX_SP2XTP4 = _kimi_k3(2560, 4, ttnn.DRAM_MEMORY_CONFIG)
# QuietBox exchange coverage: Galaxy's row count and placement with a third of its channels, still wider
# than one fabric packet per exchanged row, so every row goes out in more than one chunk.
_COVERAGE = _Geometry(640, (1024, 1024, 1024), 32, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)


def _mesh(shape: tuple[int, int], sp_axis: int, device_params: dict, topology: str, case_id: str, geometry: _Geometry):
    return pytest.param(
        shape,
        sp_axis,
        device_params,
        geometry,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology),
        id=case_id,
    )


# LoudBox, its production SP2xTP4 call on a 1D fabric (the last rank reaches rank 0 backward).
_LOUDBOX = _mesh((2, 4), 0, fabric_1d_device_params(), "mesh-2x4", "loudbox-SP2xTP4-fabric-1d", _LOUDBOX_SP2XTP4)
# Each system's ids start with it, so its CI job selects them with -k.
_MESHES = [
    _LOUDBOX,
    # QuietBox: an SP4 ring and SP2 on a 2D fabric. The ring's SP axis turns at every chip, which a 2D multicast
    # cannot follow, so it runs on a 1D ring fabric, whose routing plane follows the turns.
    _mesh(
        (1, 4),
        1,
        fabric_1d_device_params(fabric_config=ttnn.FabricConfig.FABRIC_1D_RING),
        "ring",
        "quietbox-SP4-axis1-ring",
        _COVERAGE,
    ),
    _mesh((2, 2), 0, fabric2d_device_params(), "mesh-2x2", "quietbox-SP2xTP2-fabric-2d", _COVERAGE),
]


@dataclass(frozen=True)
class _Interval:
    start: int
    # Exclusive end; None is the full capacity.
    end: int | None


def _mid_device_start(sp_size: int, local_rows: int) -> int:
    """Production's Galaxy perf gate start: half way into rank 4 (or the rank it wraps to)."""
    return 4 % sp_size * local_rows + local_rows // 2


def _intervals(sp_size: int, local_rows: int) -> list[_Interval]:
    capacity = sp_size * local_rows
    mid_device = _mid_device_start(sp_size, local_rows)
    last_rank = (sp_size - 1) * local_rows + 32
    return [
        # Rank-aligned: no split segment.
        _Interval(0, None),
        # Production's Galaxy perf gate: a full chunk starting mid-device, so its first rank also holds the tail.
        _Interval(mid_device, mid_device + capacity),
        # Bounded intervals ending mid-device and on the head of the split rank; the carry comes from the segment
        # that holds the logical end.
        _Interval(mid_device, mid_device + capacity // 2 + 64),
        _Interval(last_rank, last_rank + capacity - local_rows + 96),
        # Fewer valid rows on the final segment than the carry holds: the carry continues the previous segment, or
        # with fewer valid rows in the whole interval, the caller's history.
        _Interval(local_rows, 2 * local_rows + 2),
        _Interval(local_rows, local_rows + 2),
        # Every row of the call is valid but none is on the first rank's tail.
        _Interval(last_rank, None),
    ]


def _chronological_rows(interval: _Interval, sp_size: int, local_rows: int) -> list[tuple[int, int]]:
    """(rank, local row) of each valid row, in chronological order."""
    topology = chronological_topology(interval.start, sp_size, local_rows)
    first = topology.first_rank
    order = [(first, row) for row in range(topology.head_rows)]
    for step in range(1, sp_size):
        order.extend(((first + step) % sp_size, row) for row in range(local_rows))
    order.extend((first, row) for row in range(topology.head_rows, local_rows))
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

    def __init__(self, mesh_device: ttnn.MeshDevice, sp_axis: int, geometry: _Geometry):
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.widths = widths = geometry.widths
        self.local_rows = local_rows = geometry.local_rows
        trailing = geometry.trailing_columns
        input_memory_config = geometry.input_memory_config
        self.output_memory_config = geometry.output_memory_config
        mesh_shape = tuple(mesh_device.shape)
        self.sp_size = mesh_shape[sp_axis]
        self.other_size = mesh_shape[1 - sp_axis]
        channels = sum(widths)
        generator = torch.Generator().manual_seed(613)
        # [rank, row, channel]; replicated across the other mesh axis, whose ranks form independent SP rings.
        self.segments = torch.randn(self.sp_size, local_rows, channels, generator=generator, dtype=torch.bfloat16)
        self.history = torch.randn(1, _HISTORY_ROWS, channels, generator=generator, dtype=torch.bfloat16)
        self.taps = tuple(torch.randn(1, 1, channels, generator=generator, dtype=torch.bfloat16) for _ in range(4))
        projected = torch.cat(
            (
                self.segments.reshape(1, self.sp_size * local_rows, channels),
                torch.randn(1, self.sp_size * local_rows, trailing, generator=generator, dtype=torch.bfloat16),
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
            memory_config=input_memory_config,
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
            memory_config=self.output_memory_config,
        )

    def to_host(self, outputs: tuple[ttnn.Tensor, ...]) -> list[list[torch.Tensor]]:
        """Per device, in mesh order: the concatenated Q, K and V, and the carry."""
        q, k, v, carry = (ttnn.get_device_tensors(output) for output in outputs)
        return [
            [torch.cat([ttnn.to_torch(part) for part in parts], dim=-1)[0], ttnn.to_torch(final)]
            for *parts, final in zip(q, k, v, carry, strict=True)
        ]

    def check(self, interval: _Interval, host: list[list[torch.Tensor]]) -> None:
        order = _chronological_rows(interval, self.sp_size, self.local_rows)
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


@pytest.mark.parametrize(
    "mesh_device,sp_axis,device_params,geometry", _MESHES, indirect=["mesh_device", "device_params"]
)
def test_qkv_convolution_exchange_matches_single_sequence(
    mesh_device: ttnn.MeshDevice, sp_axis: int, device_params: dict, geometry: _Geometry
) -> None:
    """Accuracy against the single-sequence reference; determinism over back-to-back calls and trace replay."""
    exchange = _Exchange(mesh_device, sp_axis, geometry)
    try:
        for interval in _intervals(exchange.sp_size, geometry.local_rows):
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


# Traced wall time of one call, from the median of the samples.
_PERF_CALLS_PER_TRACE = 20
_PERF_SAMPLES = 7
_PERF_MARGIN = 0.05
# Galaxy SP8xTP4's per-chip call, 640 rows and 24 heads with the projection and Q, K and V in L1, on the SP ring
# each system's KDA layer perf gate runs: SP8xTP1 on LoudBox's 8 chips and SP4xTP1 on QuietBox's 4.
_GALAXY_PER_CHIP = _kimi_k3(640, 4, ttnn.L1_MEMORY_CONFIG)


def _ring_perf_case(shape: tuple[int, int], sp_axis: int, case_id: str, reference_us: float):
    return pytest.param(
        shape,
        sp_axis,
        fabric_1d_device_params(fabric_config=ttnn.FabricConfig.FABRIC_1D_RING),
        _GALAXY_PER_CHIP,
        reference_us,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology="ring"),
        id=case_id,
    )


_PERF_CASES = [
    _ring_perf_case((8, 1), 0, "loudbox-SP8xTP1-ring", 63.6),
    _ring_perf_case((4, 1), 0, "quietbox-SP4xTP1-ring", 64.0),
]


@pytest.mark.parametrize(
    "mesh_device,sp_axis,device_params,geometry,reference_us", _PERF_CASES, indirect=["mesh_device", "device_params"]
)
def test_qkv_convolution_exchange_kimi_k3_perf(
    mesh_device: ttnn.MeshDevice, sp_axis: int, device_params: dict, geometry: _Geometry, reference_us: float
) -> None:
    exchange = _Exchange(mesh_device, sp_axis, geometry)
    capacity = exchange.sp_size * geometry.local_rows
    start = _mid_device_start(exchange.sp_size, geometry.local_rows)
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
        logger.info(f"qkv_causal_conv1d_silu_with_carry Kimi-K3 perf: {median_us:.1f} us per call, {samples_us}")
        lower, upper = reference_us * (1 - _PERF_MARGIN), reference_us * (1 + _PERF_MARGIN)
        assert lower <= median_us <= upper, (
            f"{median_us:.1f} us per call outside [{lower:.1f}, {upper:.1f}] "
            f"(reference {reference_us} us, margin +/- {_PERF_MARGIN * 100:.0f}%)"
        )
    finally:
        for tensor in (actual_start, actual_end):
            ttnn.deallocate(tensor)
        exchange.deallocate()
