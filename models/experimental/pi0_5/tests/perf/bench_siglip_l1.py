# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone SigLIP (vision-stage) L1-residency benchmark for a single
BH-Galaxy island (4x4 = 16 chips). The full 8x4 GLX pipeline needs 28 chips in
one parent mesh; this host is wired as two isolated 4x4 islands, so the parent
mesh cannot form. But the SigLIP vision stage only needs a (1,4) = 4-chip
submesh, which fits trivially in one island.

This harness:
  - opens a 4x4 mesh (one island),
  - carves a (1,4) vision submesh + 4 per-chip 1x1 submeshes,
  - builds the 4-chip StageVision SigLIP pipeline,
  - times the forward with PI0_GLX_SIGLIP_L1 off vs on,
  - validates PCC vs the torch reference.

Run:
    source models/experimental/pi0_5/local_env.sh
    PI05_GLX_TRANSPORT=host \
      python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_l1.py

Env:
  PI0_GLX_SIGLIP_L1   0/1   migrate SigLIP matmul weights to L1 (the thing under test)
  BENCH_WARMUP        int   warmup iters (default 2)
  BENCH_ITERS         int   timed iters (default 10)
  PI0_NUM_CAMERAS     int   batch (default 3)
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as TorchSigLIPVisionTower
from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as TorchMMProjector
from models.experimental.pi0_5.tt.tt_bh_glx.stage_vision import StageVision
from models.experimental.pi0_5.tt.tt_bh_glx.transport import send_via_host
from models.experimental.pi0_5.tt.tt_bh_glx._l1_migration import (
    siglip_l1_enabled,
    migrate_siglip_weights_to_l1,
)

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
WARMUP = int(os.environ.get("BENCH_WARMUP", "2"))
ITERS = int(os.environ.get("BENCH_ITERS", "10"))


class _Handles:
    """Minimal MeshHandles-compatible object for the vision stage."""

    def __init__(self, parent, vision_submesh, vision_per_chip):
        self.parent = parent
        self.vision_submesh = vision_submesh
        self.vision_per_chip = vision_per_chip
        self.prefill_per_chip = []
        self.denoise_per_chip = []


class _HostTransport:
    """Force the legacy host-bounce path (no fabric sockets needed)."""

    def send(self, src_tensor, dst_mesh, **kw):
        return send_via_host(src_tensor, dst_mesh)


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def _time_forward(stage, pixel_values, sync_mesh):
    times_ms = []
    for i in range(WARMUP + ITERS):
        ttnn.synchronize_device(sync_mesh)
        t0 = time.perf_counter()
        out = stage.run(pixel_values)
        ttnn.synchronize_device(sync_mesh)
        dt = (time.perf_counter() - t0) * 1e3
        if i >= WARMUP:
            times_ms.append(dt)
        # free the per-iter output to avoid L1/DRAM growth
        if isinstance(out, ttnn.Tensor):
            last = out
    return times_ms, last


def _time_slices(stage, pixel_values):
    """Per-slice device-synced timing to locate the bottleneck."""
    chips = stage.chips
    agg = {"embed": [], "send01": [], "layA": [], "send12": [], "layB": [], "send23": [], "tail": []}
    for i in range(WARMUP + ITERS):
        pv = pixel_values
        if isinstance(pv, torch.Tensor):
            pv = ttnn.from_torch(
                pv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=chips[0], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def step(fn, mesh):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(mesh)
            return r, (time.perf_counter() - t0) * 1e3

        h0, t = step(lambda: stage.embed_slice.forward(pv), chips[0])
        if i >= WARMUP:
            agg["embed"].append(t)
        h1, t = step(lambda: stage.transport.send(h0, chips[1]), chips[1])
        if i >= WARMUP:
            agg["send01"].append(t)
        h1, t = step(lambda: stage.layer_slice_a.forward(h1), chips[1])
        if i >= WARMUP:
            agg["layA"].append(t)
        h2, t = step(lambda: stage.transport.send(h1, chips[2]), chips[2])
        if i >= WARMUP:
            agg["send12"].append(t)
        h2, t = step(lambda: stage.layer_slice_b.forward(h2), chips[2])
        if i >= WARMUP:
            agg["layB"].append(t)
        h3, t = step(lambda: stage.transport.send(h2, chips[3]), chips[3])
        if i >= WARMUP:
            agg["send23"].append(t)
        out, t = step(lambda: stage.tail_slice.forward(h3), chips[3])
        if i >= WARMUP:
            agg["tail"].append(t)
    print("  per-slice avg ms:")
    for k, v in agg.items():
        if v:
            print(f"    {k:7s} {statistics.mean(v):7.3f}")
    return out


def main():
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    bs = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    torch.manual_seed(SEED)
    pixel_values = torch.randn(bs, 3, cfg.image_size, cfg.image_size)

    # torch reference
    ref_tower = TorchSigLIPVisionTower(cfg, loader.categorized_weights["vlm_vision"])
    ref_proj = TorchMMProjector(loader.categorized_weights["vlm_projector"])
    with torch.no_grad():
        ref_out = ref_proj.forward(ref_tower.forward(pixel_values))

    l1_on = siglip_l1_enabled()
    print(
        f"\n=== SigLIP L1 bench === PI0_GLX_SIGLIP_L1={'ON' if l1_on else 'OFF'} "
        f"bs={bs} warmup={WARMUP} iters={ITERS}"
    )

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    submeshes = []
    try:
        vision_submesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        submeshes.append(vision_submesh)
        vision_per_chip = []
        for c in range(4):
            sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c))
            vision_per_chip.append(sm)
            submeshes.append(sm)

        handles = _Handles(parent, vision_submesh, vision_per_chip)
        stage = StageVision(cfg, loader.categorized_weights, handles, transport=_HostTransport())

        if l1_on:
            migrate_siglip_weights_to_l1(stage)

        if os.environ.get("BENCH_SLICES") == "1":
            _time_slices(stage, pixel_values)
        times_ms, last = _time_forward(stage, pixel_values, vision_per_chip[3])
        out = ttnn.to_torch(last)
        pcc = _compute_pcc(ref_out, out)

        avg = statistics.mean(times_ms)
        mn, mx = min(times_ms), max(times_ms)
        sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        print(f"  shape {tuple(out.shape)}  PCC={pcc:.6f}")
        print(f"  latency_ms avg={avg:.3f} min={mn:.3f} max={mx:.3f} stddev={sd:.3f}")
        print(f"METRIC siglip_ms={avg:.4f}")
        print(f"METRIC siglip_pcc={pcc:.6f}")
        ok = pcc >= 0.997
        print(f"  PCC_OK={ok}")
        if not ok:
            raise SystemExit(f"PCC {pcc:.6f} < 0.997")
    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
