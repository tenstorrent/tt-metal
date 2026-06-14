# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEMO 1 of 2 — SigLIP vision SINGLE-SHOT latency (3 images).

What this shows: the time to run ONE 3-camera observation through the full
4-chip GLX SigLIP vision encoder, end to end, with the production-grade fast
path (per-chip device traces + fabric-socket inter-chip transport, no host
bounce). This is the "pipeline fill" latency — what you pay for the first
frame / a one-off inference.

Config (all 3 images, bs=3): input (3,3,224,224) -> output (3,256,2048).
Optimizations active: matmul LoFi, host-prep fold, SDPA q256, block-sharded
slice path, per-chip traces, fabric sockets.

Pairs with demo_siglip_continuous.py (streaming throughput).

Run:
    source models/experimental/pi0_5/local_env.sh
    PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 PI0_SIGLIP_USE_FOLD=1 \
    PI0_SIGLIP_FOLD_HOST_PREP=1 PI0_SIGLIP_SDPA_QCHUNK=256 PI0_GLX_SIGLIP_BS=1 \
    python_env/bin/python models/experimental/pi0_5/tests/perf/demo_siglip_single_shot.py
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
from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
NUM_IMAGES = 3  # FIXED: this demo is always 3 cameras
ITERS = int(os.environ.get("DEMO_ITERS", "50"))


def _pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = t1.mean(), t2.mean()
    s1, s2 = t1.std(), t2.std()
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)).item()


class _Handles:
    def __init__(self, parent, vision_submesh, vision_per_chip):
        self.parent = parent
        self.vision_submesh = vision_submesh
        self.vision_per_chip = vision_per_chip


class VisionFastPath:
    """4 per-chip device traces stitched by fabric sockets (no host bounce)."""

    def __init__(self, stage, chips):
        self.stage = stage
        self.chips = chips
        self.tr = SocketTransport()
        self.in_buf = None
        self.recv = [None, None, None]
        self.tids = [None, None, None, None]
        self.outs = [None, None, None, None]

    def _bodies(self):
        return [
            lambda: self.stage.embed_slice.forward(self.in_buf),
            lambda: self.stage.layer_slice_a.forward(self.recv[0]),
            lambda: self.stage.layer_slice_b.forward(self.recv[1]),
            lambda: self.stage.tail_slice.forward(self.recv[2]),
        ]

    def build(self, pixel_values):
        chips = self.chips
        self.in_buf = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=chips[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = self._bodies()
        h0 = b[0]()
        self.recv[0] = self.tr.allocate_recv_buffer(h0, chips[1])
        self.tr.send(h0, chips[1], out_buf=self.recv[0])
        ttnn.synchronize_device(chips[1])
        h1 = b[1]()
        self.recv[1] = self.tr.allocate_recv_buffer(h1, chips[2])
        self.tr.send(h1, chips[2], out_buf=self.recv[1])
        ttnn.synchronize_device(chips[2])
        h2 = b[2]()
        self.recv[2] = self.tr.allocate_recv_buffer(h2, chips[3])
        self.tr.send(h2, chips[3], out_buf=self.recv[2])
        ttnn.synchronize_device(chips[3])
        _ = b[3]()
        for i, (chip, body) in enumerate(zip(chips, b)):
            self.tids[i] = ttnn.begin_trace_capture(chip, cq_id=0)
            self.outs[i] = body()
            ttnn.end_trace_capture(chip, self.tids[i], cq_id=0)

    def single_shot(self, host_tile):
        """One observation, end to end. host_tile is pre-tilized (upload = 1 copy)."""
        chips = self.chips
        ttnn.copy_host_to_device_tensor(host_tile, self.in_buf)
        ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=False)
        self.tr.send(self.outs[0], chips[1], out_buf=self.recv[0])
        ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=False)
        self.tr.send(self.outs[1], chips[2], out_buf=self.recv[1])
        ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=False)
        self.tr.send(self.outs[2], chips[3], out_buf=self.recv[2])
        ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=True)
        return self.outs[3]


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

        # latency
        times = []
        for i in range(5 + ITERS):
            t0 = time.perf_counter()
            fp.single_shot(host_tile)
            ttnn.synchronize_device(chips[3])
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1e3)

        print("\n" + "=" * 60)
        print("  PI0.5 SigLIP VISION — SINGLE-SHOT LATENCY  (3 images)")
        print("=" * 60)
        print(f"  input:  3 x (3, 224, 224)   output: {tuple(out.shape)}")
        print(f"  path:   4-chip GLX, per-chip traces + fabric sockets")
        print(f"  PCC vs torch reference: {pcc:.6f}  ({'PASS' if pcc >= 0.997 else 'FAIL'})")
        print("-" * 60)
        print(
            f"  single-shot latency:  avg {statistics.mean(times):6.3f} ms"
            f"   min {min(times):6.3f} ms   stddev {statistics.stdev(times):.3f}"
        )
        print("=" * 60)
    finally:
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
