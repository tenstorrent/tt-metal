# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Score the vision encoder one layer at a time against HuggingFace.

The stage bisection narrowed a bad tower PCC to the 27-block encoder and ruled
out precision (bf16 scored the same as bfp8). This narrows it further by running
both stacks in lockstep from the same input and reporting PCC after every layer,
which separates two very different diagnoses:

* layer 0 already low  -> the block is structurally wrong (rope, norm, activation)
* layer 0 fine, drifting -> error accumulation, which is a precision story after all

It also tests permutation-equivariance head on. ``--order permuted`` feeds both
stacks the block-order tokens with block-order rotary tables; ``--order raster``
feeds HuggingFace its native raster order and permutes its output afterwards.
Those two agreeing is the assumption the whole port rests on, and disagreeing
would explain the tower PCC by itself.
"""

from __future__ import annotations

import argparse
import os

import torch

import ttnn
from models.demos.blackhole.paddleocr_vl.tt.vision.functional import preprocess, raster_position_ids, vision_rope_tables
from models.demos.blackhole.paddleocr_vl.tt.vision.model import HostEmbeddings, VisionTransformer, bucket_for
from models.demos.blackhole.paddleocr_vl.tt.vision.vision_model_config import VisionModelArgs
from models.demos.blackhole.paddleocr_vl.tt.weight_mapping import map_vision_state_dict

GOLDEN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo", "golden"))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else (a @ b).item() / d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="label_shipping")
    ap.add_argument("--order", choices=("permuted", "raster"), default="permuted")
    ap.add_argument("--layers", type=int, default=27)
    a = ap.parse_args()

    g = torch.load(os.path.join(GOLDEN_DIR, f"intermediates_{a.golden}.pt"), weights_only=False)
    grid, n = g["image_grid_thw"], int(g["image_grid_thw"].prod())

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        args = VisionModelArgs(mesh, instruct=True, max_batch_size=1, max_seq_len=2048)
        dev, host = map_vision_state_dict(args.load_state_dict(), vision_head_dim=args.head_dim, strict=True)

        emb = HostEmbeddings(args.hf_config.vision_config, host)(g["pixel_values"].to(torch.bfloat16), grid)

        pre = preprocess(grid, args.head_dim, args.spatial_merge_size, bucket=bucket_for(n), permute=True)
        perm = pre["perm"]

        # ---- HuggingFace side ------------------------------------------------
        from transformers import AutoModelForImageTextToText

        hf = AutoModelForImageTextToText.from_pretrained(args.CKPT_DIR, dtype=torch.bfloat16)
        hf.eval()
        layers = hf.model.visual.vision_model.encoder.layers

        cu = torch.tensor([0, n], dtype=torch.int32)
        if a.order == "permuted":
            hf_in = emb[perm]
            cos, sin = vision_rope_tables(raster_position_ids(grid)[perm], args.head_dim)
        else:
            hf_in = emb
            cos, sin = vision_rope_tables(raster_position_ids(grid), args.head_dim)

        ref_per_layer = []
        h = hf_in.to(torch.bfloat16)
        with torch.no_grad():
            for i in range(a.layers):
                h = layers[i](h, cu_seqlens=cu, position_embeddings=(cos.to(torch.bfloat16), sin.to(torch.bfloat16)))
                if isinstance(h, tuple):
                    h = h[0]
                ref_per_layer.append(h.float().clone())
        del hf

        # ---- TT side ---------------------------------------------------------
        tt = VisionTransformer(
            args=args,
            dtype=ttnn.bfloat8_b,
            state_dict=dev,
            weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
        )
        rot = [
            ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for t in (pre["cos"], pre["sin"])
        ]

        x = tt.prepare_input(emb[perm].float(), pre["seq_len"])
        rows = []
        for i in range(a.layers):
            x = tt.blocks[i](x, rot_mats=rot)
            got = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
            got = got[:, 0, :, : args.dim].reshape(-1, args.dim)[:n]

            ref = ref_per_layer[i]
            if a.order == "raster":
                ref = ref[perm]
            rows.append((i, pcc(ref, got)))

        print("\n========== PER-LAYER VISION PCC ==========")
        print(f"golden={a.golden}  patches={n}  order={a.order}")
        for i, p in rows:
            flag = "" if p >= 0.99 else ("  <-- first drop" if all(q >= 0.99 for _, q in rows[:i]) else "  low")
            print(f"  layer {i:2d}: {p:.6f}{flag}")
        print("==========================================")
        return 0
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
