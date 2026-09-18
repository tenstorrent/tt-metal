# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Locate where the vision tower diverges from the reference.

``probe_vision_tower.py`` scores the whole vision path in one number, which is
the gate but says nothing about *which* stage is wrong. This walks the same
input through one stage at a time and scores each against the golden, so a bad
PCC lands on a specific module:

    host embeddings  -> patch conv + resampled position table
    encoder + ln_post -> the 27 reused blocks
    projector        -> the reused PatchMerger

It also A/B tests the vision q/k RoPE permute, which is the one transform this
port applies that neither the checkpoint nor tt_transformers' vision path does
for itself, and therefore the likeliest thing to have backwards.
"""

from __future__ import annotations

import argparse
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.paddleocr_vl.tt.vision.functional import preprocess
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
    ap.add_argument("--no-qk-permute", action="store_true", help="skip the vision q/k RoPE permute")
    ap.add_argument("--dtype", choices=("bfp8", "bf16"), default="bfp8")
    a = ap.parse_args()

    g = torch.load(os.path.join(GOLDEN_DIR, f"intermediates_{a.golden}.pt"), weights_only=False)
    grid, n = g["image_grid_thw"], int(g["image_grid_thw"].prod())
    logger.info(f"golden={a.golden} grid={grid.tolist()} patches={n}")

    dtype = ttnn.bfloat8_b if a.dtype == "bfp8" else ttnn.bfloat16
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        args = VisionModelArgs(mesh, instruct=True, max_batch_size=1, max_seq_len=2048)
        # vision_head_dim=0 is the escape hatch: _to_meta_rope_format keys off the
        # name, so passing the full width makes n_heads 1 and the permute a no-op.
        dev, host = map_vision_state_dict(
            args.load_state_dict(),
            vision_head_dim=args.dim if a.no_qk_permute else args.head_dim,
            strict=True,
        )

        # ---- stage 1: host embeddings ---------------------------------------
        he = HostEmbeddings(args.hf_config.vision_config, host)
        emb = he(g["pixel_values"].to(torch.bfloat16), grid)

        from transformers import AutoModelForImageTextToText

        hf = AutoModelForImageTextToText.from_pretrained(args.CKPT_DIR, dtype=torch.bfloat16)
        hf.eval()
        with torch.no_grad():
            ref_emb = hf.model.visual.vision_model.embeddings(
                g["pixel_values"].to(torch.bfloat16).unsqueeze(0), grid_thw=grid
            )
        p_emb = pcc(ref_emb.float(), emb.float())
        logger.info(f"[1] host embeddings   pcc={p_emb:.6f}  shape={tuple(emb.shape)} vs {tuple(ref_emb.shape)}")
        del hf

        # ---- run the device tower stage by stage ----------------------------
        tt = VisionTransformer(args=args, dtype=dtype, state_dict=dev, weight_cache_path=args.weight_cache_path(dtype))
        pre = preprocess(grid, args.head_dim, args.spatial_merge_size, bucket=bucket_for(n), permute=True)
        perm = pre["perm"]

        rot = []
        for t in (pre["cos"], pre["sin"]):
            rot.append(
                ttnn.from_torch(
                    t.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            )

        x = tt.prepare_input(emb[perm].float(), pre["seq_len"])
        for blk in tt.blocks:
            x = blk(x, rot_mats=rot)
        x = x[:, :, :n, :]
        x = tt.ln_post(x)

        tower_tt = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        tower_tt = tower_tt[:, 0, :, : args.dim].reshape(-1, args.dim)[:n]
        p_tower = pcc(g["tower_out"][perm], tower_tt)
        logger.info(f"[2] encoder + ln_post pcc={p_tower:.6f}")

        merged = tt.patch_merger(x)
        out_dim = args.hf_config.vision_config.out_hidden_size
        merged_t = ttnn.to_torch(merged, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        merged_t = merged_t[:, 0, :, :out_dim].reshape(-1, out_dim)[: n // (args.spatial_merge_size**2)]
        p_proj = pcc(g["projector_out"], merged_t)
        logger.info(f"[3] projector         pcc={p_proj:.6f}")

        # A reference merger run on the golden tower isolates merger from encoder.
        ref_merged_in = tt.prepare_input(g["tower_out"][perm].float(), n)
        ref_merged = tt.patch_merger(ref_merged_in)
        rm = ttnn.to_torch(ref_merged, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        rm = rm[:, 0, :, :out_dim].reshape(-1, out_dim)[: n // (args.spatial_merge_size**2)]
        p_merger_only = pcc(g["projector_out"], rm)

        print("\n============ VISION BISECTION ============")
        print(f"golden              : {a.golden}  ({n} patches)")
        print(f"qk rope permute     : {'OFF' if a.no_qk_permute else 'ON'}")
        print(f"weight dtype        : {a.dtype}")
        print(f"[1] host embeddings : {p_emb:.6f}")
        print(f"[2] encoder+ln_post : {p_tower:.6f}")
        print(f"[3] projector (e2e) : {p_proj:.6f}")
        print(f"[4] merger alone    : {p_merger_only:.6f}   (golden tower -> our merger)")
        print("==========================================")
        return 0
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
