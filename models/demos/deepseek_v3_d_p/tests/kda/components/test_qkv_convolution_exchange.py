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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    decode_tail,
    encode_tail,
    make_actual_start,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import kda_device_params
from models.demos.deepseek_v3_d_p.tt.kda.kda import _HistoryExchange
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS

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
# One 4x4 prefill stage, SP4xTP4: a 5120-token chunk is 1280 rows per SP rank, at LoudBox's per-device width.
_GALAXY_4X4_SP4XTP4 = _kimi_k3(1280, 4, ttnn.DRAM_MEMORY_CONFIG)
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


# LoudBox, its production SP2xTP4 call on a 2D fabric.
_LOUDBOX = _mesh((2, 4), 0, kda_device_params((2, 4)), "mesh-2x4", "loudbox-SP2xTP4-fabric-2d", _LOUDBOX_SP2XTP4)
# Each system's ids start with it, so its CI job selects them with -k.
_MESHES = [
    _LOUDBOX,
    # QuietBox: an SP4 ring on TorusX and SP2 on an unwrapped 2D fabric.
    _mesh((1, 4), 1, kda_device_params((1, 4)), "ring", "quietbox-SP4-axis1-ring", _COVERAGE),
    _mesh((2, 2), 0, kda_device_params((2, 2)), "mesh-2x2", "quietbox-SP2xTP2-fabric-2d", _COVERAGE),
    # One 4x4 prefill stage (a Galaxy tray half): the SP4 ring is mesh axis 1, a straight X wrap.
    _mesh((4, 4), 1, kda_device_params((4, 4)), "mesh-4x4", "galaxy4x4-SP4xTP4-torus-x", _GALAXY_4X4_SP4XTP4),
    _mesh(
        (4, 4),
        1,
        kda_device_params((4, 4), wrap_both_axes=True),
        "mesh-4x4",
        "galaxy4x4-SP4xTP4-torus-xy",
        _GALAXY_4X4_SP4XTP4,
    ),
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
    order = chronological_topology(interval.start, sp_size, local_rows).chronological_rows()
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
            encode_tail(self.history, KDA_CONVOLUTION_PAGE_CHANNELS),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
        )
        self.taps_tt = tuple(
            ttnn.from_torch(
                tap, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate
            )
            for tap in self.taps
        )
        self.exchange = _HistoryExchange.allocate(mesh_device, channels)
        # The KDA layer's compute config (FP32 DST), which its QKV convolution call runs.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def run(self, actual_start: ttnn.Tensor, actual_end: ttnn.Tensor | None) -> tuple[ttnn.Tensor, ...]:
        return ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
            self.projected,
            self.history_tt,
            *self.taps_tt,
            *self.widths,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_CHANNEL_CHUNK_SIZE),
            compute_kernel_config=self.compute_kernel_config,
            actual_start=actual_start,
            exchange_buffer=self.exchange.buffer,
            exchange_semaphores=self.exchange.semaphores,
            actual_end=actual_end,
            sequence_parallel_axis=self.sp_axis,
            memory_config=self.output_memory_config,
            page_channels=KDA_CONVOLUTION_PAGE_CHANNELS,
        )

    def to_host(self, outputs: tuple[ttnn.Tensor, ...]) -> list[list[torch.Tensor]]:
        """Per device, in mesh order: the concatenated Q, K and V, and the carry as ``[1, K-1, W]`` rows."""
        q, k, v, carry = (ttnn.get_device_tensors(output) for output in outputs)
        return [
            [
                torch.cat([ttnn.to_torch(part) for part in parts], dim=-1)[0],
                decode_tail(ttnn.to_torch(final), KDA_CONVOLUTION_PAGE_CHANNELS),
            ]
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
        kda_device_params(shape),
        _GALAXY_PER_CHIP,
        reference_us,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology="ring"),
        id=case_id,
    )


_PERF_CASES = [
    _ring_perf_case((8, 1), 0, "loudbox-SP8xTP1-ring", 58.7),
    _ring_perf_case((4, 1), 0, "quietbox-SP4xTP1-ring", 58.8),
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


# --- SP-sharded history slab -----------------------------------------------------------------------------------------
#
# The model's slab stores, per SP rank, only that rank's slice of every Q/K/V branch of each (slot, layer) entry. The
# op sends the ranks' slices of the old tail to the chronologically first rank over the fabric and writes back only
# each rank's slice of the replacement. Contract: Q, K, V and the replacement equal the replicated-history call's bit
# for bit, for every first rank, a request head (start 0: zeros), a short final segment that continues the old tail,
# and a slot past the slab (zeros, slab untouched); entries other than the addressed one stay untouched.

_SLAB_LAYERS = 2
_SLAB_SLOTS = 2
# The addressed entry is slot * _SLAB_LAYERS + _SLAB_POSITION.
_SLAB_POSITION = 1

_SHARDED_MESHES = [
    _mesh((4, 4), 1, kda_device_params((4, 4)), "mesh-4x4", "galaxy4x4-SP4xTP4-torus-x", _GALAXY_4X4_SP4XTP4),
    _mesh(
        (8, 4),
        0,
        kda_device_params((8, 4)),
        "mesh-8x4",
        "galaxy8x4-SP8xTP4-torus-xy",
        _kimi_k3(640, 4, ttnn.DRAM_MEMORY_CONFIG),
    ),
    # An SP line (the fabric does not wrap): the gather to the first rank and the exchange route along a line.
    _mesh(
        (8, 4),
        0,
        fabric2d_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
        "mesh-8x4",
        "galaxy8x4-SP8xTP4-fabric-2d",
        _kimi_k3(640, 4, ttnn.DRAM_MEMORY_CONFIG),
    ),
]


def _rank_slices(tail_pages: torch.Tensor, widths: tuple[int, int, int], sp_size: int, rank: int) -> torch.Tensor:
    """A channel-paged tail ``[..., W / P, 3P]``'s pages that SP rank ``rank`` stores: its slice of each branch."""
    parts, first = [], 0
    for width in widths:
        pages = width // KDA_CONVOLUTION_PAGE_CHANNELS
        own = pages // sp_size
        parts.append(tail_pages[..., first + rank * own : first + (rank + 1) * own, :])
        first += pages
    return torch.cat(parts, dim=-2)


class _ShardedSlab:
    """A sharded history slab of ``_SLAB_SLOTS * _SLAB_LAYERS`` entries of random tails, and its host copy."""

    def __init__(self, exchange: _Exchange, seed: int):
        self.exchange = exchange
        mesh_device = exchange.mesh_device
        channels = sum(exchange.widths)
        generator = torch.Generator().manual_seed(seed)
        entries = _SLAB_SLOTS * _SLAB_LAYERS
        # [entry, 3, W] rows of every entry's tail.
        self.tails = torch.randn(entries, _HISTORY_ROWS, channels, generator=generator, dtype=torch.bfloat16)
        pages = encode_tail(self.tails, KDA_CONVOLUTION_PAGE_CHANNELS)
        stored = torch.cat(
            [_rank_slices(pages, exchange.widths, exchange.sp_size, rank) for rank in range(exchange.sp_size)]
        )
        dims = (0, None) if exchange.sp_axis == 0 else (None, 0)
        self.slab = ttnn.from_torch(
            stored,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape)),
        )

    def run(self, slot: ttnn.Tensor, actual_start: ttnn.Tensor, actual_end: ttnn.Tensor | None):
        exchange = self.exchange
        return ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
            exchange.projected,
            self.slab,
            *exchange.taps_tt,
            *exchange.widths,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_CHANNEL_CHUNK_SIZE),
            compute_kernel_config=exchange.compute_kernel_config,
            actual_start=actual_start,
            exchange_buffer=exchange.exchange.buffer,
            exchange_semaphores=exchange.exchange.semaphores,
            actual_end=actual_end,
            sequence_parallel_axis=exchange.sp_axis,
            memory_config=exchange.output_memory_config,
            page_channels=KDA_CONVOLUTION_PAGE_CHANNELS,
            slab_index=slot,
            slab_index_scale=_SLAB_LAYERS,
            slab_index_offset=_SLAB_POSITION,
        )

    def device_entries(self) -> list[torch.Tensor]:
        """Per device in mesh order, its stored slab ``[entries, W / (P SP), 3P]``."""
        return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(self.slab)]

    def expected_entries(self, sp_rank: int) -> torch.Tensor:
        return _rank_slices(
            encode_tail(self.tails, KDA_CONVOLUTION_PAGE_CHANNELS), self.exchange.widths, self.exchange.sp_size, sp_rank
        )

    def sp_rank_of(self, device: int) -> int:
        columns = tuple(self.exchange.mesh_device.shape)[1]
        return (device // columns, device % columns)[self.exchange.sp_axis]


def _replicated_run(exchange: _Exchange, history: torch.Tensor, actual_start, actual_end) -> list[list[torch.Tensor]]:
    """The replicated-history call on ``history`` ``[1, 3, W]``: per device, Q|K|V and the carry rows."""
    history_tt = ttnn.from_torch(
        encode_tail(history, KDA_CONVOLUTION_PAGE_CHANNELS),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=exchange.mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(exchange.mesh_device),
    )
    saved, exchange.history_tt = exchange.history_tt, history_tt
    try:
        outputs = exchange.run(actual_start, actual_end)
        host = exchange.to_host(outputs)
        _release(outputs)
    finally:
        exchange.history_tt = saved
        ttnn.deallocate(history_tt)
    return host


def _sharded_cases(sp_size: int, local_rows: int) -> list[tuple[int, _Interval]]:
    """(slot, interval): every first rank (mid-segment and segment-aligned starts), the shared interval set, a request
    head, and a slot past the slab."""
    capacity = sp_size * local_rows
    cases = []
    for rank in range(sp_size):
        start = rank * local_rows + local_rows // 2
        cases.append((rank % _SLAB_SLOTS, _Interval(start, start + capacity)))
        cases.append(((rank + 1) % _SLAB_SLOTS, _Interval(rank * local_rows, None)))
    cases += [(1, interval) for interval in _intervals(sp_size, local_rows)]
    cases.append((_SLAB_SLOTS, _Interval(local_rows + 32, local_rows + 32 + capacity)))
    return cases


@pytest.mark.parametrize(
    "mesh_device,sp_axis,device_params,geometry", _SHARDED_MESHES, indirect=["mesh_device", "device_params"]
)
def test_qkv_convolution_sharded_slab_matches_replicated(
    mesh_device: ttnn.MeshDevice, sp_axis: int, device_params: dict, geometry: _Geometry
) -> None:
    """Every case one call, then a burst of back-to-back calls over changing first ranks and entries, then a traced
    call replayed on two slots: each against the replicated-history call, bit for bit, slab included."""
    exchange = _Exchange(mesh_device, sp_axis, geometry)
    slab = _ShardedSlab(exchange, seed=8461)
    zeros = torch.zeros_like(exchange.history)
    entries = _SLAB_SLOTS * _SLAB_LAYERS

    def expect(slot: int, interval: _Interval):
        """The replicated reference for one call on the slab's current host copy; updates the copy."""
        entry = slot * _SLAB_LAYERS + _SLAB_POSITION
        in_slab = entry < entries
        history = zeros if interval.start == 0 or not in_slab else slab.tails[entry : entry + 1]
        start = make_actual_start(mesh_device, interval.start)
        end = None if interval.end is None else make_actual_start(mesh_device, interval.end)
        reference = _replicated_run(exchange, history, start, end)
        for tensor in (start, end):
            if tensor is not None:
                ttnn.deallocate(tensor)
        if in_slab:
            slab.tails[entry] = reference[0][1][0]
        return reference

    def check(label: str, reference, outputs) -> None:
        q, k, v = (ttnn.get_device_tensors(output) for output in outputs[:3])
        for device, parts in enumerate(zip(q, k, v, strict=True)):
            actual = torch.cat([ttnn.to_torch(part) for part in parts], dim=-1)[0]
            assert_bit_identical(reference[device][0], actual, name=f"{label} Q, K, V, device {device}")

    def check_slab(label: str) -> None:
        for device, stored in enumerate(slab.device_entries()):
            assert_bit_identical(
                slab.expected_entries(slab.sp_rank_of(device)), stored, name=f"{label} slab, device {device}"
            )

    try:
        check_slab("upload")
        cases = _sharded_cases(exchange.sp_size, geometry.local_rows)
        for slot, interval in cases:
            label = f"slot {slot} {interval}"
            reference = expect(slot, interval)
            slot_tt = make_actual_start(mesh_device, slot)
            start = make_actual_start(mesh_device, interval.start)
            end = None if interval.end is None else make_actual_start(mesh_device, interval.end)
            outputs = slab.run(slot_tt, start, end)
            check(label, reference, outputs)
            _release(outputs[:3])
            check_slab(label)
            for tensor in (slot_tt, start, end):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            logger.info(f"sharded slab {label}: PASS")

        # Back to back with no host round trip: first ranks and entries change every call, so a call that took an
        # earlier call's arrivals, or wrote another call's entry, fails.
        burst = [(rank % (_SLAB_SLOTS + 1), cases[2 * rank][1]) for rank in range(exchange.sp_size)]
        references = [expect(slot, interval) for slot, interval in burst]
        scalars = [
            (
                make_actual_start(mesh_device, slot),
                make_actual_start(mesh_device, interval.start),
                make_actual_start(mesh_device, interval.end),
            )
            for slot, interval in burst
        ]
        results = [slab.run(*scalar) for scalar in scalars]
        for (slot, interval), reference, outputs in zip(burst, references, results, strict=True):
            check(f"burst slot {slot} {interval}", reference, outputs)
            _release(outputs[:3])
        check_slab("burst")
        for scalar in scalars:
            for tensor in scalar:
                ttnn.deallocate(tensor)
        logger.info("sharded slab back-to-back burst: PASS")

        # One capture, replayed on two slots: the slot is read on device.
        slot_tt = make_actual_start(mesh_device, 0)
        interval = cases[2][1]
        start = make_actual_start(mesh_device, interval.start)
        end = make_actual_start(mesh_device, interval.end)
        # An eager call first (it advances slot 0's entry, as expect() records), then the capture, which runs nothing.
        expect(0, interval)
        _release(slab.run(slot_tt, start, end)[:3])
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outputs = slab.run(slot_tt, start, end)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        try:
            for slot in (1, 0):
                reference = expect(slot, interval)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        torch.tensor([slot], dtype=torch.int64),
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    ),
                    slot_tt,
                )
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                check(f"trace replay slot {slot}", reference, outputs)
                check_slab(f"trace replay slot {slot}")
        finally:
            ttnn.release_trace(mesh_device, trace)
            _release(outputs[:3])
            for tensor in (slot_tt, start, end):
                ttnn.deallocate(tensor)
        logger.info("sharded slab trace replay over slots: PASS")
    finally:
        ttnn.deallocate(slab.slab)
        exchange.deallocate()
