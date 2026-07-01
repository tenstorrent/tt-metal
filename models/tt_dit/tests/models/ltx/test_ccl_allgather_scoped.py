# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scoped AllGather capture for the LTX stage-2 DiT block on mesh bh_2x4sp1tp0.

Profiling the full 928-op block under --collect-noc-traces overflows the profiler DRAM
marker buffer, so NoC bytes come out undercounted. This harness runs ONLY the block's
AllGather collective in isolation — small enough to keep the marker buffer clean — so
bandwidth = NoC-bytes / op-duration is trustworthy.

On the 2x4sp1tp0 Linear-topology path (tp_factor=2) the block's self-attn / cross-attn /
FFN inputs are TP-gathered on the feature dim before each matmul; the SP gather is fused
into ring SDPA and never appears as a standalone AllGatherAsync. This reproduces that TP
gather at the stage-2 video sequence length.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params

# Scale the gathered sequence length to probe the fabric-BW ceiling: a small transfer is
# latency/overhead-bound and understates the link rate, so LTX_CCL_SEQ_SCALE=N grows the
# per-device shard N-fold to push the link toward saturation (used to settle 200 vs 400 Gbps).
_SEQ_SCALE = int(os.environ.get("LTX_CCL_SEQ_SCALE", "1"))

# LTX-2.3-22B distilled video config (mirrors test_transformer_ltx.py).
DIM = 4096
TILE = 32

# Stage-2 1080p fast-pipeline latent grid (latent_frames, h//32, w//32).
STAGE2_F, STAGE2_H, STAGE2_W = 19, 34, 60

# NoC-trace markers share one ~11.7k-marker/core DRAM buffer. A single stage-2-shaped
# AllGather emits ~5.8k markers/core, so exactly one op fits with margin; two ops overflow,
# drop markers, and corrupt the fabric-event stream. Capture one op only — a collective's
# device-zone duration and byte count are deterministic, so no host-side warmup is needed.
WARMUP_ITERS = 0
MEASURED_ITERS = 1


def _sp_pad_len(n_real: int, sp_factor: int) -> int:
    divisor = TILE * sp_factor
    return ((n_real + divisor - 1) // divisor) * divisor


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, id="2x4sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_ccl_allgather_scoped(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
) -> None:
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    assert tp_factor > 1, "AllGather only fires for tp_factor > 1"

    video_N = _sp_pad_len(STAGE2_F * STAGE2_H * STAGE2_W * _SEQ_SCALE, sp_factor)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # Block input layout: (1, 1, N, DIM) sharded seq on SP and feature on TP. The per-device
    # shard is (1, 1, N/sp, DIM/tp); the gather collects the feature dim across the TP axis.
    spatial = torch.randn(1, 1, video_N, DIM, dtype=torch.float32)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    logger.info(
        f"AllGather scoped: per-device in {tuple(tt_spatial.shape)} bf16, dim=3, "
        f"mesh_axis={tp_axis} (tp={tp_factor}), num_links={num_links}, topology={topology}"
    )

    for _ in range(WARMUP_ITERS):
        ccl_manager.all_gather_persistent_buffer(tt_spatial, dim=3, mesh_axis=tp_axis)
    ttnn.synchronize_device(mesh_device)

    for _ in range(MEASURED_ITERS):
        out = ccl_manager.all_gather_persistent_buffer(tt_spatial, dim=3, mesh_axis=tp_axis)
    ttnn.synchronize_device(mesh_device)

    assert out.shape[3] == DIM, f"gathered feature dim {out.shape[3]} != {DIM}"
    logger.info(f"AllGather scoped done: gathered out {tuple(out.shape)}")
