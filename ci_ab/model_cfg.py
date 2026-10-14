#!/usr/bin/env python3
"""Scratch, not for merge. Which decode SDPA grid the tt_transformers model config picks per (model, batch, TP),
with a stand in device of the job's grid; no device is opened."""
import sys

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

gx, gy = (int(v) for v in sys.argv[1].split("x"))


class Grid:
    x, y = gx, gy


class Dev:
    def compute_with_storage_grid_size(self):
        return Grid()


class Args:
    pass


for name, batch, n_kv, tp in (("l8b", 1, 8, 1), ("l8b", 8, 8, 1), ("l8b", 16, 8, 1), ("l8b", 32, 8, 1),
                              ("l70b_tp4", 8, 8, 4), ("l70b_tp4", 32, 8, 4)):
    a = Args()
    a.mesh_device, a.max_batch_size, a.n_kv_heads, a.cluster_shape = Dev(), batch, n_kv, [1, tp]
    a.sdpa_decode_q_chunk_size = a.sdpa_decode_k_chunk_size = 0
    try:
        cfg = ModelArgs.get_attn_sdpa_decode_program_config.__wrapped__(a, None)
        g = cfg.compute_with_storage_grid_size
        print(f"MODEL {name} b{batch} tp{tp} grid {g.x}x{g.y} sub_core_grids {cfg.sub_core_grids is not None}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"MODEL {name} b{batch} tp{tp} error {type(e).__name__}: {str(e)[:200]}", flush=True)
