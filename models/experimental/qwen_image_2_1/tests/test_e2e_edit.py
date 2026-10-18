# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Editing path through the pipeline.

test_edit_denoise_with_golden_encoders: golden text embeddings + golden condition latents -> prefix -> 40-step
    trace -> VAE decode; latents vs the diffusers goldens (tests the DiT/pipeline part without the encoders).
test_edit_full_chain: real vision-conditioned text encoder + VAE encoder.

    python -m pytest models/experimental/qwen_image_2_1/tests/test_e2e_edit.py -v -s
"""
import json
import os
import time

import pytest
import torch
from PIL import Image

from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline

EDIT = os.path.join(GOLDENS_DIR, "edit")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(EDIT, "denoise_edit.pt")), reason="edit goldens missing"
)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _pixel_pcc(img, ref_path):
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype=np.float32).ravel()
    b = np.asarray(Image.open(ref_path).convert("RGB").resize(img.size), dtype=np.float32).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_edit_denoise_with_golden_encoders():
    te = torch.load(os.path.join(EDIT, "text_encoder_edit.pt"), weights_only=False)
    dn = torch.load(os.path.join(EDIT, "denoise_edit.pt"), weights_only=False)
    ve = torch.load(os.path.join(EDIT, "vae_encode.pt"), weights_only=False)
    dev = open_device()
    try:
        pipe = QwenImage21Pipeline(dev, load_text_encoder=False, load_vae=True, load_editing=False)
        slot_mask = te["image_pad_mask"][0].bool()
        cond_lat = ve["latent_normalized"][0, :, 0].reshape(64, -1).t().contiguous().to(torch.bfloat16)
        target_hw = (64, 64)
        t0 = time.time()
        pipe._set_target(target_hw)
        ps = pipe.prepare_prompt_with_images(te["prompt_embeds"][0], slot_mask, cond_lat, [(1, 64, 64)], target_hw)
        print(f"\nprefix (P={ps.text_len}): {time.time()-t0:.2f}s")
        n = dn["num_steps"]
        lat0 = pipe.initial_latents(dn["seed"], target_hw)
        assert torch.equal(
            lat0, dn["steps"][0]["hidden_states"][0:1, -4096:]
        ), "target noise differs from the reference"
        lat = pipe.denoise_on_device(ps, lat0, n)
        t0 = time.time()
        lat = pipe.denoise_on_device(ps, lat0, n)
        dt = time.time() - t0
        p = pcc(lat, dn["latents_after_step"][n - 1][2])
        print(
            f"edit denoise {n} steps (trace replay): {dt:.2f}s = {dt/n*1e3:.0f} ms/step; latents pcc vs golden = {p:.5f}"
        )
        rgb, rgba = pipe.decode(lat)
        rgb.save(os.path.join(EDIT, "tt_edit_golden_encoders.png"))
        print(f"pixel pcc vs reference edit image: {_pixel_pcc(rgb, os.path.join(EDIT, 'image_edit.png')):.4f}")
        pipe.release_traces()
        assert p >= 0.99
    finally:
        close_device(dev)


def test_edit_full_chain():
    dn = torch.load(os.path.join(EDIT, "denoise_edit.pt"), weights_only=False)
    meta = json.load(open(os.path.join(EDIT, "meta.json")))
    dev = open_device()
    try:
        pipe = QwenImage21Pipeline(dev)
        img = Image.open(meta["image"][0])
        t0 = time.time()
        rgb, rgba, lat, tm = pipe.generate(meta["prompt"], seed=meta["seed"], num_steps=meta["steps"], images=[img])
        print(f"\nfull edit chain: {json.dumps(tm.as_dict())}  wall {time.time()-t0:.1f}s")
        rgb.save(os.path.join(EDIT, "tt_edit_full.png"))
        p = pcc(lat, dn["latents_after_step"][meta["steps"] - 1][2])
        print(
            f"latents pcc vs golden = {p:.5f}; pixel pcc vs reference edit = {_pixel_pcc(rgb, os.path.join(EDIT, 'image_edit.png')):.4f}"
        )
        rgb2, _, lat2, tm2 = pipe.generate(meta["prompt"], seed=meta["seed"], num_steps=meta["steps"], images=[img])
        print(f"second edit (warm): {json.dumps(tm2.as_dict())}")
        pipe.release_traces()
        assert torch.equal(lat, lat2), "repeated request changed latents"
        assert p >= 0.98
    finally:
        close_device(dev)
