# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash image front end on ONE chip: the ttnn ViT + aligner (``tt/v41/vision.py``) on each image, through the
checkpoint's own ``image_processor.load_image``, written as an aligner-rows file per image that tt-blaze's
``prof/vit_prompt.py --rows`` turns into a ring prompt (virtual ids + span rows; tt-blaze vit_plan.md V1 / V3).

    python models/demos/deepseek_v3_d_p/scripts/v41_vit_rows.py --image a.jpg [--image b.jpg ...] --out-dir <dir>

Writes ``<dir>/<image stem>.rows.pt`` = {"dev_rows": [n_llm_h * n_llm_w, 5120] bf16, "image", "grid" (ViT h, w),
"llm_grid", "vit_ms", "aligner_ms"}. The weights are uploaded once; the first image also compiles (its timing is the
second, warm run). No CPU reference: the accuracy of record is ``tests/pcc/test_v41_vision.py`` (DS41F-0045).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.config import DEFAULT_MODEL_DIR
from models.demos.deepseek_v3_d_p.tt.v41.vision import V41Vision, load_vision_state
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(a.model_dir, "inference"))
    import image_processor as ip  # noqa: E402 -- the checkpoint's own preprocessing

    args = types.SimpleNamespace(**json.load(open(os.path.join(a.model_dir, "inference", "config.json"))))
    state = load_vision_state(checkpoint(a.model_dir))
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        t0 = time.perf_counter()
        dev = V41Vision(mesh, state)
        print(f"[v41-vit-rows] weights uploaded in {time.perf_counter() - t0:.1f} s", flush=True)
        warmed = set()
        for path in a.image:
            patches, n_h, n_w, n_llm_h, n_llm_w = ip.load_image({"url": path}, args)
            if (n_h, n_w) not in warmed:  # program compile for this grid
                dev.aligner(dev.vit(patches, n_h, n_w), n_h, n_w)
                warmed.add((n_h, n_w))
            t0 = time.perf_counter()
            y = dev.vit(patches, n_h, n_w)
            t1 = time.perf_counter()
            rows = dev.aligner(y, n_h, n_w)
            t2 = time.perf_counter()
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(a.out_dir, f"{stem}.rows.pt")
            torch.save(
                dict(
                    dev_rows=rows,
                    image=path,
                    grid=(n_h, n_w),
                    llm_grid=(n_llm_h, n_llm_w),
                    vit_ms=(t1 - t0) * 1e3,
                    aligner_ms=(t2 - t1) * 1e3,
                ),
                out,
            )
            print(
                f"[v41-vit-rows] {path}: ViT grid {n_h} x {n_w} ({patches.shape[0]} patches), rows {tuple(rows.shape)}; "
                f"ViT {(t1 - t0) * 1e3:.1f} ms + aligner {(t2 - t1) * 1e3:.1f} ms -> {out}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
