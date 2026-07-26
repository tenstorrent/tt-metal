# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace profiler for the 1x8 production-payload SP affine prefix probe."""

from __future__ import annotations

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.tt.sp_affine_prefix import SP8AffinePrefixProbe

_SP_SIZE = 8
_HEADS_PER_TP4_RANK = 8
_DIM = 128

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


def _production_rank_transforms() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create the same exact 1 MiB/rank transport payload as the PCC gate."""
    generator = torch.Generator().manual_seed(8441)
    eye = torch.eye(_DIM, dtype=torch.float32).expand(_HEADS_PER_TP4_RANK, -1, -1)
    return (
        [
            (0.88 + 0.01 * span) * eye + 0.002 * torch.randn(_HEADS_PER_TP4_RANK, _DIM, _DIM, generator=generator)
            for span in range(_SP_SIZE)
        ],
        [0.01 * torch.randn(_HEADS_PER_TP4_RANK, _DIM, _DIM, generator=generator) for _ in range(_SP_SIZE)],
    )


def _device_transforms(probe: SP8AffinePrefixProbe) -> tuple[tuple[ttnn.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    host_a, host_b = _production_rank_transforms()
    return (
        tuple(
            ttnn.from_torch(
                tensor,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            for tensor, device in zip(host_a, probe.span_devices, strict=True)
        ),
        tuple(
            ttnn.from_torch(
                tensor,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            for tensor, device in zip(host_b, probe.span_devices, strict=True)
        ),
    )


def test_sp8_affine_prefix_production_payload_perf(mesh_device: ttnn.MeshDevice) -> None:
    """Measure the three-stage 1 MiB/rank prefix with independent child traces."""
    probe = SP8AffinePrefixProbe(mesh_device)
    transform_a, transform_b = _device_transforms(probe)
    # Socket binaries must be in the program cache before capture. Queue the
    # warmup without host stage barriers: those barriers use device events,
    # which are intentionally unavailable inside a trace.
    probe.run(transform_a, transform_b, retain_buffers=True, synchronize_stages=False)
    probe._synchronize()
    trace_ids = tuple(ttnn.begin_trace_capture(device, cq_id=0) for device in probe.span_devices)
    outputs = probe.run(transform_a, transform_b, retain_buffers=True, synchronize_stages=False)
    for device, trace_id in zip(probe.span_devices, trace_ids, strict=True):
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

    for device, trace_id in zip(probe.span_devices, trace_ids, strict=True):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    probe._synchronize()

    repetitions = int(os.getenv("PERF_REPS", "10"))
    signpost(header="sp8_affine_prefix_start")
    for _ in range(repetitions):
        for device, trace_id in zip(probe.span_devices, trace_ids, strict=True):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        probe._synchronize()
    signpost(header="sp8_affine_prefix_stop")

    for device, trace_id in zip(probe.span_devices, trace_ids, strict=True):
        ttnn.release_trace(device, trace_id)
    # The captured graph owns all retained intermediate buffers.  Device
    # teardown reclaims them after the profiler has emitted its report.
    del outputs
