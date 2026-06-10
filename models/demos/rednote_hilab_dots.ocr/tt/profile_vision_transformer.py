# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy / wall-clock harness for TtVisionTransformer.

Two operating points:

- default: the golden 784-patch input (gate-size doc, padded to 896 rows);
- ``--image PATH``: PRODUCTION document scale — the image is preprocessed
  with the checkpoint's Qwen2VL image processor exactly as ocr_model.py
  does (e.g. /tmp/demo_image1_cropped.jpg -> ~11k vision tokens).

``--dtype {bf16,fp32}`` selects the tower config: bf16 weights+activations
(production, optimization REDO) or the fp32 high-precision path.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_vision_transformer.py \
        --traced --dtype bf16 --image /tmp/demo_image1_cropped.jpg

Run for wall clock only:
    python models/demos/rednote_hilab_dots.ocr/tt/profile_vision_transformer.py \
        --dtype bf16 --image /tmp/demo_image1_cropped.jpg --iters 5
"""

import argparse
import importlib.util
import json
import sys
import time
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


def _load_inputs(image_path):
    """(x [seq, 588], grid_thw [1,3]) — golden tensor or HF-preprocessed image."""
    if image_path is None:
        golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_transformer.pt")
        return golden["input"], golden["grid_thw"]
    from PIL import Image
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(REPO, trust_remote_code=True)
    vis = proc(images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
    return vis["pixel_values"].float(), vis["image_grid_thw"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--image", default=None, help="production image (e.g. /tmp/demo_image1_cropped.jpg)")
    parser.add_argument("--iters", type=int, default=3, help="timed untraced iterations")
    parser.add_argument("--tp", type=int, default=1, help="head-parallel TP degree (1=replicate, 4=3 heads/chip)")
    args = parser.parse_args()
    dtype = ttnn.bfloat16 if args.dtype == "bf16" else ttnn.float32

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=400_000_000,
    )
    try:
        x, grid_thw = _load_inputs(args.image)
        seq, patch_dim = x.shape
        num_heads = 12
        spatial_merge_size = 2
        print(f"input: seq={seq} patch_dim={patch_dim} grid_thw={grid_thw.tolist()} dtype={args.dtype} tp={args.tp}")

        sd = _load_vision_tower_weights()
        model = TtVisionTransformer(mesh_device, sd, num_layers=42, num_heads=num_heads, dtype=dtype, tp_degree=args.tp)

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
            dtype=dtype,
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

        # Untraced wall-clock (the tower runs once per image in production).
        times = []
        for _ in range(max(1, args.iters)):
            t0 = time.perf_counter()
            out = model.forward(x_tt, rot_mats, cu_tt)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(out)
        times.sort()
        print(f"untraced wall ms: min={times[0]:.1f} median={times[len(times)//2]:.1f} max={times[-1]:.1f}")

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = model.forward(x_tt, rot_mats, cu_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the timed/profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            print(f"traced wall ms: {(time.perf_counter() - t0) * 1000:.1f}")
            ttnn.release_trace(mesh_device, tid)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
