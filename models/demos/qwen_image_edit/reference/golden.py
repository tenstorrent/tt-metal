# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF golden for the image_edit task: diffusers QwenImageEditPipeline in float32 on CPU.

The golden is the real pipeline __call__ with the same encoded inputs the TT pipeline uses: same images,
prompts, negative prompt, initial noise (passed as `latents`), step count and CFG scale. The only knob
changed from the pipeline default is the condition-image target area (calculate_dimensions(1024*1024) is
hard-coded there), which is overridden to EditConfig.area so both sides run the same size.

Results are cached under models/demos/qwen_image_edit/_golden/<key>.pt with intermediates for diagnosis
(prompt embeds + masks, image latents, per-step latents) and the final image in [0, 1].

    python -m models.demos.qwen_image_edit.reference.golden --batch 32 --steps 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

from models.demos.qwen_image_edit.tt.inputs import MODEL_ID, EditConfig, encode_inputs

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_golden")


def golden_key(cfg: EditConfig) -> str:
    d = dict(
        b=cfg.batch, area=cfg.area, n=cfg.num_inference_steps, cfg=cfg.true_cfg_scale, neg=cfg.negative_prompt, v=2
    )
    h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
    return f"golden_b{cfg.batch}_a{int(cfg.area ** 0.5)}_n{cfg.num_inference_steps}_{h}"


def golden_path(cfg: EditConfig) -> str:
    return os.path.join(GOLDEN_DIR, golden_key(cfg) + ".pt")


def load_hf_pipeline(dtype=torch.float32):
    from diffusers import QwenImageEditPipeline

    return QwenImageEditPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)


def _hf_reference_image_edit(cfg: EditConfig, pipe=None, enc=None, threads: int | None = None):
    """Run the HF QwenImageEditPipeline and return the golden dict."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit as edit_mod

    if threads:
        torch.set_num_threads(threads)
    enc = enc if enc is not None else encode_inputs(cfg)
    pipe = pipe if pipe is not None else load_hf_pipeline()

    rec = {"step_latents": []}
    orig_calc = edit_mod.calculate_dimensions
    orig_encode = pipe.encode_prompt
    orig_prepare = pipe.prepare_latents

    def calc(target_area, ratio):
        return orig_calc(cfg.area, ratio)

    def encode_prompt(*a, **k):
        pe, pm = orig_encode(*a, **k)
        rec.setdefault("prompt_embeds", []).append(pe.clone())
        rec.setdefault("prompt_masks", []).append(None if pm is None else pm.clone())
        return pe, pm

    def prepare_latents(*a, **k):
        lat, img_lat = orig_prepare(*a, **k)
        rec["image_latents"] = img_lat.clone()
        rec["init_latents"] = lat.clone()
        return lat, img_lat

    def on_step(p, i, t, kw):
        rec["step_latents"].append(kw["latents"].clone())
        return {}

    edit_mod.calculate_dimensions = calc
    pipe.encode_prompt = encode_prompt
    pipe.prepare_latents = prepare_latents
    try:
        t0 = time.time()
        with torch.no_grad():
            out = pipe(
                image=enc.images,
                prompt=enc.prompts,
                negative_prompt=[cfg.negative_prompt] * cfg.batch,
                true_cfg_scale=cfg.true_cfg_scale,
                num_inference_steps=cfg.num_inference_steps,
                latents=enc.latents.clone(),
                output_type="pt",
                callback_on_step_end=on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        elapsed = time.time() - t0
    finally:
        edit_mod.calculate_dimensions = orig_calc
        pipe.encode_prompt = orig_encode
        pipe.prepare_latents = orig_prepare

    return {
        "image": out.images.to(torch.float32),  # [B, 3, H, W] in [0, 1]
        "prompt_embeds": rec["prompt_embeds"][0],
        "prompt_mask": rec["prompt_masks"][0],
        "neg_prompt_embeds": rec["prompt_embeds"][1] if len(rec["prompt_embeds"]) > 1 else None,
        "neg_prompt_mask": rec["prompt_masks"][1] if len(rec["prompt_masks"]) > 1 else None,
        "image_latents": rec["image_latents"],
        "init_latents": rec["init_latents"],
        "step_latents": torch.stack(rec["step_latents"]),
        "width": enc.width,
        "height": enc.height,
        "prompts": enc.prompts,
        "seeds": enc.seeds,
        "cfg": dict(cfg.__dict__),
        "seconds": elapsed,
    }


def load_or_build_golden(cfg: EditConfig, build_if_missing: bool = False):
    path = golden_path(cfg)
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    if not build_if_missing:
        return None
    g = _hf_reference_image_edit(cfg)
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    torch.save(g, path)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--area", type=int, default=256)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    cfg = EditConfig(batch=a.batch, area=a.area * a.area, num_inference_steps=a.steps, true_cfg_scale=a.cfg_scale)
    torch.set_num_threads(a.threads)
    path = golden_path(cfg)
    print("golden ->", path, flush=True)
    g = _hf_reference_image_edit(cfg)
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    torch.save(g, path)
    print(f"done in {g['seconds']:.1f}s image {tuple(g['image'].shape)}", flush=True)


if __name__ == "__main__":
    main()
