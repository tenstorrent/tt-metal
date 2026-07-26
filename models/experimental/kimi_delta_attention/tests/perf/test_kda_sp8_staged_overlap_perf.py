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
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
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
