# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtVisionTransformer at the PRODUCTION operating point.

Production context (tests/test_vision_transformer.py): the full fp32 vision
tower (patch_embed -> 42x block -> post_trunk_norm -> merger) runs once per
image on a replicated [1, 1, 896, 588] pre-flattened patch tensor on the 1x4
mesh (784 patches padded to 896 rows, hidden 1536, 12 heads x head_dim 128,
windowed SDPA over cu_seqlens), with ALL real fp32 vision_tower weights.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_vision_transformer.py --traced
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn

_TT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _TT_DIR.parent
REPO = "rednote-hilab/dots.ocr"

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_vision_transformer", _TT_DIR / "vision_transformer.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtVisionTransformer = _mod.TtVisionTransformer

_ref_spec = importlib.util.spec_from_file_location("dots_ocr_ref_functional", MODEL_DIR / "reference" / "functional.py")
_ref = importlib.util.module_from_spec(_ref_spec)
sys.modules[_ref_spec.name] = _ref
_ref_spec.loader.exec_module(_ref)


def _load_vision_tower_weights():
    """All vision_tower.* tensors from the HF checkpoint, prefix stripped."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    prefix = "vision_tower."
    out = {}
    by_file = {}
    for full, fname in idx.items():
        if full.startswith(prefix):
            by_file.setdefault(fname, []).append(full)
    for fname, keys in by_file.items():
        with safe_open(snap / fname, framework="pt") as f:
            for full in keys:
                out[full[len(prefix) :]] = f.get_tensor(full).float()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=400_000_000,
    )
    try:
        golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_transformer.pt")
        x, grid_thw = golden["input"], golden["grid_thw"]
        seq, patch_dim = x.shape
        num_heads = 12
        spatial_merge_size = 2

        sd = _load_vision_tower_weights()
        # Production dtype: fp32 residual stream + fp32 weights (see tt module docs).
        model = TtVisionTransformer(mesh_device, sd, num_layers=42, num_heads=num_heads, dtype=ttnn.float32)

        head_dim = sd["blocks.0.attn.qkv.weight"].shape[-1] // num_heads
        rope = _ref.vision_rot_pos_emb(grid_thw, head_dim=head_dim, spatial_merge_size=spatial_merge_size)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        padded_seq = ((seq + 127) // 128) * 128
        x_pad = torch.cat([x, torch.zeros(padded_seq - seq, patch_dim)], dim=0)

        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x_pad.reshape(1, 1, padded_seq, patch_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_mats = model.prepare_rope(rope, padded_seq)
        cu_tt = model.prepare_cu_seqlens(cu_seqlens)

        # Warmup: compile every kernel into the program cache.
        for _ in range(2):
            out = model.forward(x_tt, rot_mats, cu_tt)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = model.forward(x_tt, rot_mats, cu_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = model.forward(x_tt, rot_mats, cu_tt)
            ttnn.synchronize_device(mesh_device)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
