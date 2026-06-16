# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone SigLIP perf A/B: single-chip (current) vs camera-parallel DP.

Both paths run the IDENTICAL full SigLIP encoder (SigLIPCameraSlice = embed + 27
blocks + post_ln + projector) on the same weights/input, differing only in HOW the
3-camera batch is parallelized:

  - single : all 3 cameras on ONE chip at bs=3   (matches the single-chip SigLIP in
             test_perf_ttnn_full_e2e_trace.py — full encoder, one device)
  - dp     : 1 camera per chip on 3 chips, run concurrently, then gather

Reports traced replay latency for each + speedup + PCC(dp vs single). Camera-parallel
is math-identical (each image is independent), so PCC ~1.0 is expected.

Run (needs full pi0.5 ckpt + 28-chip galaxy):
  PI05_CHECKPOINT_DIR=<ckpt> PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/pytest -s \
    models/experimental/pi0_5/tests/perf/test_perf_siglip_dp_vs_single.py
"""
import os

# Per-head parallelism for the head TM ops (C++ getenv at op-build time, so set
# before any device op is constructed). Without these, NlpCreateQkvHeads /
# NlpConcatHeads decompose work per-sequence-tile only (batch*seq/32 = 24 cores
# at bs=3), leaving most of the grid idle. With them, create splits by num_kv
# heads (-> full 120-core grid) and concat splits into 8 head-groups. Matches
# the production config used by the e2e tests.
os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
# _prefix_setup() uploads the pre-folded (B, H, W/patch, C*patch) layout, which only
# PatchEmbeddingTTNN._forward_fold accepts. That fast path is gated on PI0_SIGLIP_USE_FOLD;
# without it forward() falls into the generic unfold path and the reshape volume mismatches.
os.environ.setdefault("PI0_SIGLIP_USE_FOLD", "1")

import statistics
import time
from pathlib import Path

import pytest
import torch
import ttnn

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
NCAM = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
NPERF = int(os.environ.get("PERF_ITERS", "20"))

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _prefix_setup(imgs_bchw, device, patch=14):
    """Host-side prefix_setup (matches PI0_SIGLIP_USE_FOLD production path): permute
    BCHW→BHWC and pre-reshape to (B, H, W/patch, C*patch) on host, upload ROW_MAJOR.
    Lets PatchEmbeddingTTNN._forward_fold take its fast path (b) — the device skips the
    permute/reshape/untilize (~0.89 ms) it would otherwise do on a raw TILE input."""
    x = imgs_bchw.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
    B, H, W, C = x.shape
    x = x.reshape(B, H, W // patch, C * patch).contiguous()  # (B,H,W/patch,C*patch)
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _time_traced(submeshes, fwds, sync_dev, nperf):
    """Capture one trace per submesh, replay all concurrently, return avg ms."""
    tids = [ttnn.begin_trace_capture(sm, cq_id=0) for sm in submeshes]
    for fwd in fwds:
        fwd()
    for sm, tid in zip(submeshes, tids):
        ttnn.end_trace_capture(sm, tid, cq_id=0)
    ttnn.synchronize_device(sync_dev)
    for _ in range(3):  # warmup
        for sm, tid in zip(submeshes, tids):
            ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
        for sm in submeshes:
            ttnn.synchronize_device(sm)
    ts = []
    for _ in range(nperf):
        s = time.perf_counter()
        for sm, tid in zip(submeshes, tids):
            ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
        for sm in submeshes:
            ttnn.synchronize_device(sm)
        ts.append((time.perf_counter() - s) * 1000)
    for sm, tid in zip(submeshes, tids):
        ttnn.release_trace(sm, tid)
    return statistics.mean(ts), min(ts), max(ts)


def test_perf_siglip_single_vs_dp():
    from models.experimental.pi0_5.common.configs import SigLIPConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
    from models.experimental.pi0_5.tt.tt_bh_glx.vision_slice import SigLIPCameraSlice

    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    vw = loader.categorized_weights["vlm_vision"]
    pw = loader.categorized_weights["vlm_projector"]
    torch.manual_seed(SEED)
    px = torch.randn(NCAM, 3, cfg.image_size, cfg.image_size)

    print(f"\n--- SigLIP A/B: single-chip bs={NCAM} vs camera-parallel DP ({NCAM} chips) ---")
    with open_galaxy_mesh(l1_small_size=24576) as h:
        chips = h.vision_per_chip
        assert len(chips) >= NCAM, f"need >={NCAM} vision chips, got {len(chips)}"

        # Full SigLIP on each of the first NCAM chips (same weights replicated).
        cams = [SigLIPCameraSlice(cfg, vw, pw, chips[i]) for i in range(NCAM)]

        # ---- SINGLE-CHIP: all cameras on chips[0] at bs=NCAM ----
        # prefix_setup on host (matches production fold path → fast _forward_fold).
        px_single = _prefix_setup(px, chips[0], patch=cfg.patch_size)
        out_single = ttnn.to_torch(cams[0].forward(px_single))  # (NCAM,256,2048)
        single_ms = _time_traced([chips[0]], [lambda: cams[0].forward(px_single)], chips[0], NPERF)

        # ---- DP: 1 camera per chip, concurrent, then gather to chips[0] ----
        from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport

        transport = SocketTransport()
        # prefix_setup per camera on its own chip (1 camera each).
        px_dp = [_prefix_setup(px[i : i + 1], chips[i], patch=cfg.patch_size) for i in range(NCAM)]
        # EAGER warmup BEFORE capture: the first forward builds lazy BS-memcfg / LN
        # program-config caches (host writes) and the first transport.send creates the
        # socket pairs + recv buffers (host writes) — both are forbidden during trace
        # capture. Run them once eagerly here; this also yields the PCC reference.
        outs_e = [cams[i].forward(px_dp[i]) for i in range(NCAM)]
        recvd_e = [transport.send(outs_e[i], chips[0]) for i in range(1, NCAM)]
        gathered_e = ttnn.concat([outs_e[0], *recvd_e], dim=0)
        ttnn.synchronize_device(chips[0])
        out_dp = ttnn.to_torch(gathered_e)  # (NCAM,256,2048) on chips[0]

        # Capture compute-only traces (caches + sockets now warm); keep output handles.
        tids = [ttnn.begin_trace_capture(chips[i], cq_id=0) for i in range(NCAM)]
        outs_t = [cams[i].forward(px_dp[i]) for i in range(NCAM)]
        for i in range(NCAM):
            ttnn.end_trace_capture(chips[i], tids[i], cq_id=0)
        ttnn.synchronize_device(chips[NCAM - 1])

        def _replay_and_gather():
            for i in range(NCAM):
                ttnn.execute_trace(chips[i], tids[i], cq_id=0, blocking=False)
            # gather: fabric-send cams 1..N-1 → chips[0] (sockets pre-established), concat
            recvd = [transport.send(outs_t[i], chips[0]) for i in range(1, NCAM)]
            return ttnn.concat([outs_t[0], *recvd], dim=0)

        for _ in range(3):  # warmup replays
            _replay_and_gather()
            ttnn.synchronize_device(chips[0])
        dts = []
        for _ in range(NPERF):
            s = time.perf_counter()
            _replay_and_gather()
            ttnn.synchronize_device(chips[0])
            dts.append((time.perf_counter() - s) * 1000)
        for i in range(NCAM):
            ttnn.release_trace(chips[i], tids[i])
        dp_ms = (statistics.mean(dts), min(dts), max(dts))

        pcc = _pcc(out_dp, out_single)
        print("\n" + "=" * 64)
        print(f"  SigLIP single-chip (bs={NCAM}, 1 chip) : {single_ms[0]:6.2f} ms  (min {single_ms[1]:.2f})")
        print(f"  SigLIP camera-parallel DP ({NCAM} chips): {dp_ms[0]:6.2f} ms  (min {dp_ms[1]:.2f})  [incl gather]")
        print(f"  speedup                          : {single_ms[0] / dp_ms[0]:5.2f}x")
        print(f"  PCC(DP vs single-chip)           : {pcc:.6f}")
        print("=" * 64)

        assert pcc > 0.99, f"DP PCC {pcc:.6f} < 0.99 — should be math-identical"
        assert dp_ms[0] < single_ms[0], "DP should be faster than single-chip bs=N"
