# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eager stage-barrier overlap probe for the SP=8 KDA affine prefix.

LoudBox cannot run a literal SP=8 x TP=4 layer.  This test instead queues a
TP=4-rank-shaped local KDA grouped scan on each of its eight chips while the
first real prefix stage's socket transfers are in flight.  Crucially, the
prefix performs its all-device stage barrier before it composes or starts the
next distance.  The scan release is intentionally optimistic (all ranks rather
than only ranks whose entries are ready), so this is a stability/capacity probe
and not end-to-end SP8 correctness.
"""

from __future__ import annotations

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention, PreparedKDA
from models.experimental.kimi_delta_attention.tt.sp_affine_prefix import SP8AffinePrefixProbe


pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]


def _production_rank_transforms(
    probe: SP8AffinePrefixProbe,
) -> tuple[tuple[ttnn.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    generator = torch.Generator().manual_seed(9017)
    transform_a = []
    transform_b = []
    for device in probe.span_devices:
        a = torch.eye(128, dtype=torch.float32).repeat(8, 1, 1)
        a = a + 0.002 * torch.randn((8, 128, 128), generator=generator)
        b = 0.02 * torch.randn((8, 128, 128), generator=generator)
        transform_a.append(
            ttnn.from_torch(
                a,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        transform_b.append(
            ttnn.from_torch(
                b,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    return tuple(transform_a), tuple(transform_b)


def _prepare_rank_scans(
    probe: SP8AffinePrefixProbe, sequence: int
) -> tuple[
    tuple[KimiDeltaAttention, ...],
    tuple[PreparedKDA, ...],
    tuple[list[ttnn.Tensor], ...],
    tuple[ttnn.Tensor, ...],
]:
    """Prepare the local eight-head scan work for each physical LB device."""
    config = KDAConfig(
        hidden_size=2304,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(9119)).to(
        torch.bfloat16
    )
    layers = tuple(KimiDeltaAttention(device, config, random_weights(config)) for device in probe.span_devices)
    prepared = []
    grouped = []
    entries = []
    for layer, device in zip(layers, probe.span_devices, strict=True):
        layer.reset_state(batch_size=1)
        span = ttnn.from_torch(
            hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rank_prepared = layer.prepare_chunk(span)
        rank_grouped = layer.group_prepare(rank_prepared)
        transform_a, transform_b = layer.group_summary(rank_grouped)
        prepared.append(rank_prepared)
        grouped.append(rank_grouped)
        entries.append(layer.group_entry_states(transform_a, transform_b, sequence // 256))
    probe._synchronize()
    return layers, tuple(prepared), tuple(grouped), tuple(entries)


def _enqueue_scans(
    layers: tuple[KimiDeltaAttention, ...],
    prepared: tuple[PreparedKDA, ...],
    grouped: tuple[list[ttnn.Tensor], ...],
    entries: tuple[ttnn.Tensor, ...],
) -> tuple[ttnn.Tensor, ...]:
    """Queue the real local grouped scan; its inputs are already device-resident."""
    outputs = []
    for layer, rank_prepared, rank_grouped, rank_entries in zip(layers, prepared, grouped, entries, strict=True):
        output, final_state = layer.group_scan(rank_prepared, rank_grouped, rank_entries, output_final_state=False)
        assert final_state is None
        outputs.append(output)
    return tuple(outputs)


def _execute_stage_traces(
    probe: SP8AffinePrefixProbe,
    stage_traces: tuple[tuple[int, ...], ...],
    repetitions: int,
) -> None:
    """Replay each distance on all ranks, then establish its global boundary."""
    for _ in range(repetitions):
        for trace_ids in stage_traces:
            for device, trace_id in zip(probe.span_devices, trace_ids, strict=True):
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            probe._synchronize()


def _stage_sliced_trace_overlap(
    probe: SP8AffinePrefixProbe,
    transform_a: tuple[ttnn.Tensor, ...],
    transform_b: tuple[ttnn.Tensor, ...],
    layers: tuple[KimiDeltaAttention, ...],
    prepared: tuple[PreparedKDA, ...],
    grouped: tuple[list[ttnn.Tensor], ...],
    entries: tuple[ttnn.Tensor, ...],
    repetitions: int,
) -> None:
    """Capture one trace per rank and prefix distance, with host stage fences."""
    # All program binaries must be in cache before capture. These retained
    # outputs also keep the allocated stage addresses stable for trace capture.
    warm_scan_outputs = _enqueue_scans(layers, prepared, grouped, entries)
    warm_a, warm_b = probe.run(transform_a, transform_b, retain_buffers=True, synchronize_stages=True)
    probe._synchronize()

    prefix_a, prefix_b = transform_a, transform_b
    stage_traces = []
    trace_outputs: list[ttnn.Tensor] = []
    for stage in range(3):
        trace_ids = tuple(ttnn.begin_trace_capture(device, cq_id=0) for device in probe.span_devices)
        scans: tuple[ttnn.Tensor, ...] = ()

        def enqueue_stage_zero_scan(current_stage: int) -> None:
            nonlocal scans
            if current_stage == 0:
                scans = _enqueue_scans(layers, prepared, grouped, entries)

        prefix_a, prefix_b = probe.run_stage(
            prefix_a,
            prefix_b,
            stage,
            retain_buffers=True,
            after_enqueued=enqueue_stage_zero_scan,
        )
        for device, trace_id in zip(probe.span_devices, trace_ids, strict=True):
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        stage_traces.append(trace_ids)
        trace_outputs.extend(scans)

    _execute_stage_traces(probe, tuple(stage_traces), 1)
    signpost(header="sp8_stage_sliced_trace_start")
    _execute_stage_traces(probe, tuple(stage_traces), repetitions)
    signpost(header="sp8_stage_sliced_trace_stop")

    for device, trace_ids in zip(probe.span_devices, zip(*stage_traces, strict=True), strict=True):
        for trace_id in trace_ids:
            ttnn.release_trace(device, trace_id)


def test_sp8_stage_barrier_overlap_stability(mesh_device: ttnn.MeshDevice) -> None:
    """Verify the stage-barrier schedule drains fabric under rank-shaped work."""
    sequence = int(os.getenv("PERF_LOCAL_SEQ", "2560"))
    if sequence % 256:
        raise ValueError(f"PERF_LOCAL_SEQ must be divisible by 256, got {sequence}")
    probe = SP8AffinePrefixProbe(mesh_device)
    transform_a, transform_b = _production_rank_transforms(probe)
    layers, prepared, grouped, entries = _prepare_rank_scans(probe, sequence)
    scan_outputs: tuple[ttnn.Tensor, ...] = ()

    def enqueue_after_stage_zero(stage: int) -> None:
        nonlocal scan_outputs
        if stage == 0:
            scan_outputs = _enqueue_scans(layers, prepared, grouped, entries)

    if os.getenv("PERF_STAGE_SLICED_TRACE", "0") == "1":
        _stage_sliced_trace_overlap(
            probe,
            transform_a,
            transform_b,
            layers,
            prepared,
            grouped,
            entries,
            int(os.getenv("PERF_REPS", "3")),
        )
    else:
        signpost(header="sp8_stage_barrier_overlap_start")
        prefix_a, prefix_b = probe.run(
            transform_a,
            transform_b,
            synchronize_stages=True,
            after_stage_enqueued=enqueue_after_stage_zero,
        )
        signpost(header="sp8_stage_barrier_overlap_stop")
        for output in scan_outputs:
            ttnn.deallocate(output)
        for tensor in (*prefix_a, *prefix_b):
            ttnn.deallocate(tensor)
