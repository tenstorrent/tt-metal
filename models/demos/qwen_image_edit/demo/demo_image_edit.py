# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit image editing on a T3K (Call 1: image_edit).

Loads real images + edit instructions, encodes them with the HF processors, runs the chained TT
pipeline (tt/pipeline.py: build_pipeline -> run_image_edit, the same code the e2e test runs) and
writes the edited images as PNGs.

    python -m models.demos.qwen_image_edit.demo.demo_image_edit                      # 32 bundled samples
    python -m models.demos.qwen_image_edit.demo.demo_image_edit \\
        --image models/sample_data/huggingface_cat_image.jpg --prompt "Give the cat a blue wizard hat."
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from PIL import Image

from models.demos.qwen_image_edit.mesh import close_mesh, open_mesh
from models.demos.qwen_image_edit.tt import pipeline as P
from models.demos.qwen_image_edit.tt.inputs import EditConfig, sample_images, sample_prompts, sample_seeds


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="*", default=None, help="input image path(s); default: the bundled samples")
    ap.add_argument("--prompt", nargs="*", default=None, help="edit instruction(s), one per image (or one for all)")
    ap.add_argument("--batch", type=int, default=32, help="samples per call when using the bundled samples")
    ap.add_argument("--steps", type=int, default=50, help="num_inference_steps (pipeline default 50)")
    ap.add_argument("--area", type=int, default=256, help="target side: images are resized to ~area x area")
    ap.add_argument("--cfg-scale", type=float, default=4.0, help="true_cfg_scale (model card: 4.0)")
    ap.add_argument("--negative-prompt", default=" ")
    ap.add_argument("--seed", type=int, default=None, help="base seed (default: 1000, one per sample)")
    ap.add_argument("--out", default="models/demos/qwen_image_edit/demo/output", help="output directory")
    ap.add_argument("--compare-golden", action="store_true", help="print PCC vs the cached HF golden if present")
    return ap.parse_args(argv)


def _inputs(a):
    if a.image:
        images = [Image.open(p).convert("RGB") for p in a.image]
        prompts = a.prompt or ["Make it look like a watercolor painting."]
        if len(prompts) == 1:
            prompts = prompts * len(images)
        assert len(prompts) == len(images), "give one prompt, or one per image"
        # one calculated size per call: centre-crop every image to a square
        sq = []
        for im in images:
            s = min(im.size)
            x0, y0 = (im.size[0] - s) // 2, (im.size[1] - s) // 2
            sq.append(im.crop((x0, y0, x0 + s, y0 + s)))
        images = sq
    else:
        images, prompts = sample_images(a.batch), sample_prompts(a.batch)
    n = len(images)
    seeds = [(a.seed if a.seed is not None else 1000) + i for i in range(n)]
    if not a.image and a.seed is None:
        seeds = sample_seeds(n)
    # the VAE runs batch-parallel over the 2 mesh rows: pad an odd batch with a copy (dropped on output)
    pad = n % 2
    if pad:
        images, prompts, seeds = images + images[-1:], prompts + prompts[-1:], seeds + seeds[-1:]
    return images, prompts, seeds, n


def main(argv=None):
    a = parse_args(argv)
    images, prompts, seeds, n_real = _inputs(a)
    cfg = EditConfig(
        batch=len(images),
        area=a.area * a.area,
        num_inference_steps=a.steps,
        true_cfg_scale=a.cfg_scale,
        negative_prompt=a.negative_prompt,
    )
    os.makedirs(a.out, exist_ok=True)
    hf = P.load_hf_reference(torch.float32)
    mesh = open_mesh()
    try:
        t0 = time.time()
        pipe = P.build_pipeline(mesh, model=hf, cfg=cfg)
        print(f"built pipeline in {time.time() - t0:.1f}s", flush=True)
        enc = pipe.encode(cfg, images=images, prompts=prompts, seeds=seeds)
        p = pipe.prepare(enc)
        t0 = time.time()
        out = pipe.run_image_edit(p)
        image = P.to_host(out).to(torch.float32)
        dt = time.time() - t0
        print(
            f"edited {n_real} image(s) at {enc.width}x{enc.height}, {cfg.num_inference_steps} steps in {dt:.1f}s",
            flush=True,
        )
        torch.save(
            {"image": image[:n_real], "prompts": prompts[:n_real], "seeds": seeds[:n_real]},
            os.path.join(a.out, "images.pt"),
        )
        for i in range(n_real):
            arr = (image[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            path = os.path.join(a.out, f"edit_{i:02d}.png")
            Image.fromarray(arr).save(path)
            enc.images[i].save(os.path.join(a.out, f"input_{i:02d}.png"))
            print(f"  [{i:02d}] '{prompts[i]}' -> {path}", flush=True)
        if a.compare_golden and not a.image:
            from models.common.utility_functions import comp_pcc
            from models.demos.qwen_image_edit.reference.golden import load_or_build_golden

            g = load_or_build_golden(cfg)
            if g is None:
                print("no cached golden for this config", flush=True)
            else:
                pccs = [float(comp_pcc(g["image"][b], image[b], 0.99)[1]) for b in range(n_real)]
                print(f"PCC vs HF golden: min {min(pccs):.6f} mean {sum(pccs) / len(pccs):.6f}", flush=True)
                print(f"e2e PCC={min(pccs)}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
