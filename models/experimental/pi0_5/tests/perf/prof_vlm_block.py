# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile ONE Gemma-2B VLM block (GLX prefill) on a 1x1 submesh, to quantify
the matmul-fidelity (LoFi) opportunity for the VLM prefill stage — mirroring
the SigLIP analysis. VLM dims: width=2048, mlp=16384, 8 heads, head_dim=256,
seq=1024. MLP matmuls are 4x bigger than SigLIP so LoFi should help more.

Run under tracy:
    source models/experimental/pi0_5/local_env.sh
    PI0_EXPERT_MM_LOFI=0 python_env/bin/python -m tracy -v -r -p \
      -o generated/profiler/vlm_hifi2 \
      models/experimental/pi0_5/tests/perf/prof_vlm_block.py
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.tt_bh_glx.vlm_slice import VLMBlockSlice

CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream")
SEQ = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "1024"))
ITERS = int(os.environ.get("PROF_ITERS", "2"))
LAYER = int(os.environ.get("VLM_LAYER", "0"))


def main():
    from pathlib import Path

    loader = Pi0_5WeightLoader(CKPT)
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(Path(CKPT)), num_denoising_steps=5)
    vcfg = cfg.vlm_config
    weights = loader.categorized_weights["vlm_language"]

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    sm = None
    try:
        sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        slice_blk = VLMBlockSlice(vcfg, weights, sm, layer_idx=LAYER, max_seq_len=SEQ)

        torch.manual_seed(0)
        hidden = ttnn.from_torch(
            torch.randn(1, 1, SEQ, vcfg.width),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=sm,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # warmup
        out, _ = slice_blk.forward(hidden)
        ttnn.synchronize_device(sm)
        for _ in range(ITERS):
            out, _ = slice_blk.forward(hidden)
            ttnn.synchronize_device(sm)
        ttnn.ReadDeviceProfiler(sm)
        print("VLM_BLOCK_PROFILED_OK", tuple(out.shape) if hasattr(out, "shape") else "?")
    finally:
        if sm is not None:
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
