# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEMO 2 of 2 — SigLIP vision CONTINUOUS / streaming throughput (3 images).

What this shows: when observations stream continuously (a running robot
policy), the 4 vision chips PIPELINE across frames — while chip3 finishes
frame N, chip2 is already on frame N+1, etc. Steady-state per-frame time
collapses to the slowest single stage, not the serial sum.

Reported as: initial fill latency (first frame, = single-shot) PLUS the
steady-state per-frame throughput once the pipeline is full.

Config (all 3 images, bs=3): input (3,3,224,224) -> output (3,256,2048).
Same fast path as demo_siglip_single_shot.py (per-chip traces + fabric sockets).

Run:
    source models/experimental/pi0_5/local_env.sh
    PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 PI0_SIGLIP_USE_FOLD=1 \
    PI0_SIGLIP_FOLD_HOST_PREP=1 PI0_SIGLIP_SDPA_QCHUNK=256 PI0_GLX_SIGLIP_BS=1 \
    python_env/bin/python models/experimental/pi0_5/tests/perf/demo_siglip_continuous.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as TorchSigLIPVisionTower
from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as TorchMMProjector
from models.experimental.pi0_5.tt.tt_bh_glx.stage_vision import StageVision
from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport

# reuse the fast-path implementation from the single-shot demo
from models.experimental.pi0_5.tests.perf.demo_siglip_single_shot import (
    VisionFastPath,
    _Handles,
    _pcc,
)

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
NUM_IMAGES = 3  # FIXED: this demo is always 3 cameras
STREAM_FRAMES = int(os.environ.get("DEMO_STREAM_FRAMES", "60"))


def main():
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    pixel_values = torch.randn(NUM_IMAGES, 3, cfg.image_size, cfg.image_size)

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    ref_tower = TorchSigLIPVisionTower(cfg, loader.categorized_weights["vlm_vision"])
    ref_proj = TorchMMProjector(loader.categorized_weights["vlm_projector"])
    with torch.no_grad():
        ref_out = ref_proj.forward(ref_tower.forward(pixel_values))

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    try:
        vsub = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        chips = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c)) for c in range(4)]
        stage = StageVision(cfg, loader.categorized_weights, _Handles(parent, vsub, chips), transport=SocketTransport())

        fp = VisionFastPath(stage, chips)
        fp.build(pixel_values)
        host_tile = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # correctness
        out = ttnn.to_torch(fp.single_shot(host_tile))
        pcc = _pcc(ref_out, out)

        # 1) fill latency = single-shot (first frame, pipeline empty)
        fill = []
        for i in range(5 + 20):
            t0 = time.perf_counter()
            fp.single_shot(host_tile)
            ttnn.synchronize_device(chips[3])
            if i >= 5:
                fill.append((time.perf_counter() - t0) * 1e3)
        fill_ms = sum(fill) / len(fill)

        # 2) steady-state throughput: enqueue N frames back-to-back, sync once
        def stream(n):
            for _ in range(n):
                ttnn.execute_trace(chips[0], fp.tids[0], cq_id=0, blocking=False)
                fp.tr.send(fp.outs[0], chips[1], out_buf=fp.recv[0])
                ttnn.execute_trace(chips[1], fp.tids[1], cq_id=0, blocking=False)
                fp.tr.send(fp.outs[1], chips[2], out_buf=fp.recv[1])
                ttnn.execute_trace(chips[2], fp.tids[2], cq_id=0, blocking=False)
                fp.tr.send(fp.outs[2], chips[3], out_buf=fp.recv[2])
                ttnn.execute_trace(chips[3], fp.tids[3], cq_id=0, blocking=False)
            ttnn.synchronize_device(chips[3])

        stream(5)  # warm
        t0 = time.perf_counter()
        stream(STREAM_FRAMES)
        total_ms = (time.perf_counter() - t0) * 1e3
        per_frame = total_ms / STREAM_FRAMES
        fps = 1000.0 / per_frame

        print("\n" + "=" * 60)
        print("  PI0.5 SigLIP VISION — CONTINUOUS / STREAMING  (3 images)")
        print("=" * 60)
        print(f"  input:  3 x (3, 224, 224)   output: {tuple(out.shape)}")
        print(f"  path:   4-chip GLX pipeline, per-chip traces + fabric sockets")
        print(f"  PCC vs torch reference: {pcc:.6f}  ({'PASS' if pcc >= 0.997 else 'FAIL'})")
        print("-" * 60)
        print(f"  initial fill latency (frame 1):   {fill_ms:6.3f} ms")
        print(f"  steady-state throughput:          {per_frame:6.3f} ms/frame   ({fps:.0f} fps)")
        print(f"  total for {STREAM_FRAMES} frames:              {total_ms:6.1f} ms")
        print("-" * 60)
        print(f"  model:  fill {fill_ms:.2f} ms  +  {per_frame:.2f} ms x (N-1) frames")
        print("=" * 60)
    finally:
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
