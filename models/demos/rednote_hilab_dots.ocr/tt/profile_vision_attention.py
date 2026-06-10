# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtVisionAttention at the PRODUCTION operating point.

Production context (tt/ocr_model.py): the bf16 vision tower with
head-parallel TP across the 1x4 line (tp_degree=4, 3 heads/chip) calls
vision_attention 42x per image at DOCUMENT scale (~11k vision tokens for
/tmp/demo_image1_cropped.jpg), 12 heads x head_dim 128.

Operating points:

- default: bf16 + tp=4 + ``--image /tmp/demo_image1_cropped.jpg`` style
  document input (PRODUCTION, optimization REDO);
- ``--image None`` falls back to the golden 784-patch gate input (896 rows);
- ``--dtype fp32 --tp 1`` selects the legacy high-precision path used by
  tests/test_vision_transformer.py.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_vision_attention.py \
        --traced --image /tmp/demo_image1_cropped.jpg
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

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_vision_attention", _TT_DIR / "vision_attention.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtVisionAttention = _mod.TtVisionAttention

_ref_spec = importlib.util.spec_from_file_location("dots_ocr_ref_functional", MODEL_DIR / "reference" / "functional.py")
_ref = importlib.util.module_from_spec(_ref_spec)
sys.modules[_ref_spec.name] = _ref
_ref_spec.loader.exec_module(_ref)

HIDDEN = 1536
NUM_HEADS = 12
HEAD_DIM = 128
SPATIAL_MERGE = 2


def _load_weights():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in ("qkv.weight", "proj.weight"):
        full = f"vision_tower.blocks.0.attn.{k}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


def _load_grid(image_path):
    """grid_thw [1, 3] — golden gate input or HF-preprocessed document image."""
    if image_path is None:
        golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_attention.pt")
        return golden["rotary_pos_emb"], golden["cu_seqlens"], golden["cu_seqlens"][-1].item()
    from PIL import Image
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(REPO, trust_remote_code=True)
    vis = proc(images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
    grid_thw = vis["image_grid_thw"]
    rope = _ref.vision_rot_pos_emb(grid_thw, head_dim=HEAD_DIM, spatial_merge_size=SPATIAL_MERGE)
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    return rope, cu_seqlens, cu_seqlens[-1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--tp", type=int, default=4, help="head-parallel TP degree (1=replicate, 4=3 heads/chip)")
    parser.add_argument("--image", default=None, help="production image (e.g. /tmp/demo_image1_cropped.jpg)")
    parser.add_argument("--iters", type=int, default=5, help="timed untraced iterations")
    args = parser.parse_args()
    dtype = ttnn.bfloat16 if args.dtype == "bf16" else ttnn.float32

    rope, cu_seqlens, seq = _load_grid(args.image)
    padded_seq = ((seq + 127) // 128) * 128
    print(f"input: seq={seq} padded_seq={padded_seq} dtype={args.dtype} tp={args.tp}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=120_000_000,
    )
    try:
        block = TtVisionAttention(mesh_device, _load_weights(), num_heads=NUM_HEADS, dtype=dtype, tp_degree=args.tp)
        rot_mats = block.prepare_rope(rope, padded_seq)
        cu_tt = block.prepare_cu_seqlens(cu_seqlens)

        torch.manual_seed(0)
        x = torch.randn(1, 1, padded_seq, HIDDEN)
        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x,
            dtype=dtype,
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

        # Untraced wall-clock (the block runs 42x once per image in production).
        times = []
        for _ in range(max(1, args.iters)):
            t0 = time.perf_counter()
            out = block.forward(x_tt, rot_mats, cu_tt)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(out)
        times.sort()
        print(f"untraced wall ms: min={times[0]:.2f} median={times[len(times)//2]:.2f} max={times[-1]:.2f}")

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(x_tt, rot_mats, cu_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the timed/profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            print(f"traced wall ms: {(time.perf_counter() - t0) * 1000:.2f}")
            ttnn.release_trace(mesh_device, tid)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
