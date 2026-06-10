# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtVisionBlock at the PRODUCTION operating point.

Production context (tests/test_vision_transformer.py): the fp32 vision tower
calls vision_block 42x per image on a replicated [1, 1, 896, 1536] fp32
residual stream on the 1x4 mesh (784 patches padded to 896 rows, hidden 1536,
12 heads x head_dim 128, windowed SDPA over cu_seqlens).

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_vision_block.py --traced
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _TT_DIR.parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_vision_block", _TT_DIR / "vision_block.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtVisionBlock = _mod.TtVisionBlock

PADDED_SEQ = 896  # 784 real patches padded to a multiple of 128 (host-side)
HIDDEN = 1536

BLOCK_KEYS = [
    "norm1.weight",
    "attn.qkv.weight",
    "attn.proj.weight",
    "norm2.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "mlp.fc3.weight",
]


def _load_weights():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download("rednote-hilab/dots.ocr", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for short in BLOCK_KEYS:
        full = f"vision_tower.blocks.0.{short}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[short] = f.get_tensor(full).float()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=60_000_000,
    )
    try:
        # Production dtype: the vision tower runs an fp32 residual stream.
        block = TtVisionBlock(mesh_device, _load_weights(), num_heads=12, dtype=ttnn.float32)

        golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_block.pt")
        rope, cu_seqlens = golden["rotary_pos_emb"], golden["cu_seqlens"]
        rot_mats = block.prepare_rope(rope, PADDED_SEQ)
        cu_tt = block.prepare_cu_seqlens(cu_seqlens)

        torch.manual_seed(0)
        x = torch.randn(1, 1, PADDED_SEQ, HIDDEN)
        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = block.forward(x_tt, rot_mats, cu_tt)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(x_tt, rot_mats, cu_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = block.forward(x_tt, rot_mats, cu_tt)
            ttnn.synchronize_device(mesh_device)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
