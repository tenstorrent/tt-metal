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


# These tests run the RING fabric config, but the collectives have always been handed
# Topology.Linear -- the wraparound link is enabled and unused. Selectable so the two can be
# measured against each other; default stays Linear.
def _topology():
    if os.environ.get("DIFFVAE_TOPOLOGY", "linear").lower() == "ring":
        return ttnn.Topology.Ring
    return ttnn.Topology.Linear


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
def test_decode_wsp_timing(*, mesh_device, latent_hw, decode_tree):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    from models.tt_dit.parallel.manager import CCLManager

    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    # 145-frame (6s) is the target resolution -- default here so the timing test exercises the real
    # workload. output_frames = 8 * latent_T - 7, so latent_T=19 -> 145 frames; override with
    # DIFFVAE_LATENT_T (e.g. 4 -> 25 frames) for a quick smaller run.
    t_lat = int(os.environ.get("DIFFVAE_LATENT_T", 19))
    latent = torch.randn(1, config["in_channels"], t_lat, lh, lw)
    ccl = CCLManager(
        mesh_device, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)), topology=_topology()
    )  # DIFFVAE_TP_HEADS=1 adds TP-over-heads on the orthogonal (rows, size-4) mesh axis: stage-5
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
        stages_tp_axis=tp_axis if stages_wsp else None,  # 2-D SP x TP for the det stages too
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
        f"\n[decode W-SP({backend}){tp}{det} 4x8] latent(1,{config['in_channels']},{t_lat},{lh},{lw}) -> {px_shape}: {dt * 1000:8.0f} ms\n"
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
    ccl = CCLManager(mesh_device, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)), topology=_topology())
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


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# Individual PCC + runtime at 1080p 25-frame, for the no-TP sharded path and the TP+col-qkv sharded path,
# each measured against the GATHER-mesh decode as the reference (the dense-masked-attention NA3D backend --
# the highest-fidelity path that fits at 1080p; single-chip replicated OOMs at the tail). All three run on
# the same 4x8 mesh. The two sharded configs share the full lever stack (fused W-SP, det-SP, q_chunk,
# T-inner) and differ only in TP (stage-5 heads + column-parallel qkv, and TP-heads on the det stages).
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_decode_1080p_tp_pcc(*, mesh_device):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    from loguru import logger

    from models.tt_dit.parallel.manager import CCLManager

    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, 34, 60)  # 1080p, 25 frames
    ccl = CCLManager(mesh_device, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)), topology=_topology())

    def timed(dec):
        dec.decode(latent, seed=0)  # warmup
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        px = dec.decode(latent, seed=0)
        ttnn.synchronize_device(mesh_device)
        return px.float(), (time.perf_counter() - t0) * 1000.0

    def sharded(tp_axis, tp_proj):
        os.environ["DIFFVAE_SP_FUSED"] = "1"
        os.environ["DIFFVAE_TP_PROJ"] = "1" if tp_proj else "0"
        dec = DiffVAEDecoder(
            config,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            stage5_na3d_backend="op_sp_w_sharded",
            stage5_sp_axis=1,
            stage5_tp_axis=tp_axis,
            stages_na3d_backend="op_sp_w_sharded",
            stages_sp_axis=1,
            stages_tp_axis=tp_axis,
        )
        dec.load_checkpoint(CHECKPOINT)
        return timed(dec)

    # Reference: gather-mesh decode (dense masked attention, query-sharded across the mesh).
    ref_dec = DiffVAEDecoder(
        config, mesh_device=mesh_device, ccl_manager=ccl, stages_na3d_backend="gather", stage5_na3d_backend="gather"
    )
    ref_dec.load_checkpoint(CHECKPOINT)
    ref, t_ref = timed(ref_dec)

    no_tp, t_no = sharded(None, False)
    tp_full, t_tp = sharded(0, True)

    logger.info(f"[1080p-pcc] gather-mesh reference:          runtime {t_ref:8.0f} ms")
    logger.info(f"[1080p-pcc] no-TP sharded:  PCC {_pcc(no_tp, ref) * 100:.4f} %   runtime {t_no:8.0f} ms")
    logger.info(f"[1080p-pcc] TP + col-qkv:   PCC {_pcc(tp_full, ref) * 100:.4f} %   runtime {t_tp:8.0f} ms")
