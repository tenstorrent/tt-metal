# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S2 probe: score the vision tower against the HuggingFace reference.

Runs the real patch tensors recorded by ``generate_goldens.py`` through the
Blackhole tower and compares the projected image embeddings to what HF produced
from the same input. Those embeddings are what gets spliced into the text
stream, so this is the gate that decides whether the vision half is correct.

The comparison is on the projector output rather than the raw tower output,
because our tower deliberately emits tokens in merge-block order while HF's is
raster. The projector output is order-identical between the two by construction
(see ``tests/test_vision_permutation.py``), which makes it the natural meeting
point. ``--stage tower`` permutes the golden and compares the encoder stack
alone, for bisecting a failure.

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    python models/demos/blackhole/paddleocr_vl/tests/probe_vision_tower.py
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.paddleocr_vl.tt.vision.model import DropInVisionTransformer, bucket_for
from models.demos.blackhole.paddleocr_vl.tt.vision.vision_model_config import VisionModelArgs
from models.demos.blackhole.paddleocr_vl.tt.weight_mapping import map_vision_state_dict, summarize

GOLDEN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo", "golden"))
PCC_TARGET = 0.98


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (a @ b).item() / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("projector", "tower"), default="projector")
    ap.add_argument("--only", default=None, help="restrict to one golden by name")
    args_cli = ap.parse_args()

    goldens = sorted(glob.glob(os.path.join(GOLDEN_DIR, "intermediates_*.pt")))
    if args_cli.only:
        goldens = [g for g in goldens if args_cli.only in g]
    if not goldens:
        logger.error(f"no intermediates_*.pt in {GOLDEN_DIR}; run generate_goldens.py first")
        return 2
    logger.info(f"scoring against {len(goldens)} golden(s)")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        model_args = VisionModelArgs(mesh, instruct=True, max_batch_size=1, max_seq_len=2048)
        device_sd, host_sd = map_vision_state_dict(
            model_args.load_state_dict(), vision_head_dim=model_args.head_dim, strict=True
        )
        logger.info(summarize(device_sd, host_sd))

        tower = DropInVisionTransformer(
            model_args=model_args,
            device_state_dict=device_sd,
            host_state_dict=host_sd,
            dtype=ttnn.bfloat8_b,
        )
        logger.info("tower built")

        results = []
        for path in goldens:
            name = os.path.basename(path).removeprefix("intermediates_").removesuffix(".pt")
            g = torch.load(path, weights_only=False)
            grid = g["image_grid_thw"]
            n = int(grid.prod())
            logger.info(f"[{name}] grid={grid.tolist()} patches={n} bucket={bucket_for(n)}")

            out = tower(g["pixel_values"].to(torch.bfloat16), grid)
            ref = g["projector_out"]

            p = pcc(ref, out)
            results.append((name, n, tuple(out.shape), tuple(ref.shape), p))
            logger.info(f"[{name}] tt={tuple(out.shape)} ref={tuple(ref.shape)} pcc={p:.6f}")

        print("\n================ S2 RESULT ================")
        print(f"{'golden':22s} {'patches':>7s} {'tt shape':>14s} {'ref shape':>14s} {'PCC':>9s}  gate")
        ok = True
        for name, n, ts, rs, p in results:
            passed = ts == rs and p >= PCC_TARGET
            ok &= passed
            print(f"{name:22s} {n:7d} {str(ts):>14s} {str(rs):>14s} {p:9.6f}  {'PASS' if passed else 'FAIL'}")
        print(f"\ngate: PCC >= {PCC_TARGET} and shapes equal -> {'ALL PASS' if ok else 'FAILURES'}")
        print("===========================================")
        return 0 if ok else 1
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
