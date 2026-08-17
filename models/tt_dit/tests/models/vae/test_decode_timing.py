# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TEMP end-to-end DiffVAE decode timing (uncommitted): full replicated decode on shipped weights,
timing the whole pipeline per NA3D backend so we can see where the fused kernel places us e2e."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)


# (deterministic-stages backend, stage-5 backend). "gather+fused5" is the interesting mixed config:
# fast gather where it fits (the smaller early stages) and the memory-light fused only for stage 5,
# the stage that OOMs gather at 1080p. Set DIFFVAE_STAGE_TIMING=1 for the per-stage breakdown.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "backends",
    [("gather", "gather"), ("fused", "fused"), ("gather", "fused")],
    ids=["gather", "fused", "gather+fused5"],
)
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_timing(*, mesh_device, backends, latent_hw):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    stages_b, stage5_b = backends
    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, lh, lw)

    dec = DiffVAEDecoder(config, mesh_device=mesh_device, stages_na3d_backend=stages_b, stage5_na3d_backend=stage5_b)
    dec.load_checkpoint(CHECKPOINT)

    px = dec.decode(latent, seed=0)  # warmup (also builds fused mask cache)
    ttnn.synchronize_device(mesh_device)
    px_shape = tuple(px.shape)

    t0 = time.perf_counter()
    px = dec.decode(latent, seed=0)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    tag = f"stages={stages_b},stage5={stage5_b}"
    print(f"\n[decode {tag}] latent(1,{config['in_channels']},4,{lh},{lw}) -> {px_shape}: {dt * 1000:8.0f} ms\n")


# Stage-5 spatial-W SP across the mesh: shards Q/output over W (sp-way). DIFFVAE_SP_FUSED=1 runs the fast
# fused kernel per shard instead of the streamed op. Times op-sharded vs fused-sharded on the same mesh.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_wsp_timing(*, mesh_device, latent_hw):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    from models.tt_dit.parallel.manager import CCLManager

    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, lh, lw)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    # DIFFVAE_TP_HEADS=1 adds TP-over-heads on the orthogonal (rows, size-4) mesh axis: stage-5
    # attention runs on heads/4 of the 4 heads per chip, gathered back before the output proj.
    tp_axis = 0 if os.environ.get("DIFFVAE_TP_HEADS") == "1" else None
    # DIFFVAE_STAGES_WSP=1 also W-shards the deterministic stages (stage 0 stays replicated), so the
    # whole decode runs 1/sp instead of only stage 5 -- reclaiming the replicated det-stage time.
    stages_wsp = os.environ.get("DIFFVAE_STAGES_WSP") == "1"
    dec = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        stage5_na3d_backend="op_sp_w_sharded",
        stage5_sp_axis=1,
        stage5_tp_axis=tp_axis,
        stages_na3d_backend="op_sp_w_sharded" if stages_wsp else None,
        stages_sp_axis=1 if stages_wsp else None,
    )
    dec.load_checkpoint(CHECKPOINT)

    px = dec.decode(latent, seed=0)  # warmup
    ttnn.synchronize_device(mesh_device)
    px_shape = tuple(px.shape)
    t0 = time.perf_counter()
    px = dec.decode(latent, seed=0)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    backend = "fused" if os.environ.get("DIFFVAE_SP_FUSED") == "1" else "op"
    tp = "+TP4" if tp_axis is not None else ""
    det = "+detSP" if stages_wsp else ""
    print(
        f"\n[decode W-SP({backend}){tp}{det} 4x8] latent(1,{config['in_channels']},4,{lh},{lw}) -> {px_shape}: {dt * 1000:8.0f} ms\n"
    )


# Stage-5 with the "gather" backend on the mesh: its native NA3DShard splits QUERY tiles across all
# 32 chips (K/V stay replicated per chip), so the per-call gather -- the wall that OOMs replicated on
# ONE chip -- shrinks 32x and should fit at 25-frame 1080p, while using the faster gather executor.
# The activation stays replicated (full memory), so this is the 25-frame path, not the 6s path.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_gather_mesh_timing(*, mesh_device, latent_hw):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    from models.tt_dit.parallel.manager import CCLManager

    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, lh, lw)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    dec = DiffVAEDecoder(config, mesh_device=mesh_device, ccl_manager=ccl, stage5_na3d_backend="gather")
    dec.load_checkpoint(CHECKPOINT)

    px = dec.decode(latent, seed=0)  # warmup
    ttnn.synchronize_device(mesh_device)
    px_shape = tuple(px.shape)
    t0 = time.perf_counter()
    px = dec.decode(latent, seed=0)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    print(
        f"\n[decode gather-mesh 4x8] latent(1,{config['in_channels']},4,{lh},{lw}) -> {px_shape}: {dt * 1000:8.0f} ms\n"
    )
