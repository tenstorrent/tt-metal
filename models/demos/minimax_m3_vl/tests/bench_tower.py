# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warm wall-clock benchmark for the M3-VL vision tower (+projector).

NOT a pytest. Run:
    MESH_DEVICE=N150 TT_METAL_LOGGER_LEVEL=FATAL HF_HOME=/localdev/zbaczewski/hf_cache \
        python models/demos/minimax_m3_vl/tests/bench_tower.py
"""
import time

import torch

import ttnn
from models.demos.minimax_m3_vl.tt._m3_loader import build_reference
from models.demos.minimax_m3_vl.tt.model import M3VLVisionModel
from models.demos.minimax_m3_vl.tt.model_config import MiniMaxM3VLModelArgs

WARMUP, ITERS = 2, 10

# (tag, image side in px). side/14 = patches per side; L = (side/14)**2.
SIZES = [
    ("224", 224),  # 256
    ("448", 448),  # 1024
    ("896", 896),  # 4096
    ("1344", 1344),  # 9216
    ("1568", 1568),  # 12544
    ("1792", 1792),  # 16384
]


def tower_flops(L, d=1280, inter=5120, heads=16, hd=80, layers=32, patch_in=1176):
    """Forward MAC-counted FLOPs (2 per multiply-add). 'Useful' (unpadded) math."""
    pe = 2 * L * patch_in * d
    qkv = 3 * (2 * L * d * d)
    attn = 2 * (2 * L * L * d)  # QK^T + AV
    o = 2 * L * d * d
    mlp = 2 * (2 * L * d * inter)  # fc1 + fc2
    per_layer = qkv + attn + o + mlp
    return pe + layers * per_layer


def proj_flops(L, d=1280, ph=6144, th=6144, merged=24576):
    Lm = L // 4
    return 2 * L * d * ph + 2 * L * ph * th + 2 * Lm * merged * ph + 2 * Lm * ph * th


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    args = MiniMaxM3VLModelArgs(mesh_device=dev, dtype=ttnn.bfloat16)
    print("building reference + model (one-time)...")
    ref = build_reference(args)
    model = M3VLVisionModel.from_torch(dev, ref, args, with_projector=True, dtype=ttnn.bfloat16)

    for tag, side in SIZES:
        n = side // 14
        L = n * n
        grid = torch.tensor([[1, n, n]], dtype=torch.int64)
        pv = torch.randn(L, args.patch_flat_dim)
        gflop = (tower_flops(L) + proj_flops(L)) / 1e9
        try:
            for _ in range(WARMUP):
                out = model(pv, grid)
                ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                out = model(pv, grid)
                ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / ITERS
            print(
                f"[{tag:>4}] L={L:6d}  {gflop:8.1f} GFLOP  "
                f"latency={dt*1e3:8.2f} ms  throughput={gflop/dt/1e3:7.2f} TFLOP/s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{tag:>4}] L={L:6d}  FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

    ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
