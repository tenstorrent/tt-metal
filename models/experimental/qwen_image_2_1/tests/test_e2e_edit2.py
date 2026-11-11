# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two condition images (the editing demo: llama.jpg + gongnyang.jpg -> "happy, jumping together and hooraying").

test_edit2_golden_encoders: golden text embeddings + golden condition latents through the DiT/pipeline (2 image blocks).
test_edit2_full_chain: real vision text encoder + VAE encoder end to end; saves the demo image.

    python -m pytest models/experimental/qwen_image_2_1/tests/test_e2e_edit2.py -v -s
"""
import json
import os
import time

import pytest
import torch
from PIL import Image

import ttnn
from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline

EDIT = os.path.join(GOLDENS_DIR, "edit2")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(EDIT, "denoise_edit.pt")), reason="edit2 goldens missing"
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


def _layout(te, dn, ve):
    slot_mask = te["image_pad_mask"][0].bool()
    img_shapes = [tuple(x) for x in dn["steps"][0]["img_shapes"][0]]
    target = img_shapes[-1]
    cond_lat = (
        torch.cat([e["latent_normalized"][0, :, 0].reshape(64, -1).t() for e in ve["all_images"]], 0)
        .contiguous()
        .to(torch.bfloat16)
    )
    return slot_mask, img_shapes, (target[1], target[2]), cond_lat


def test_edit2_golden_encoders():
    te = torch.load(os.path.join(EDIT, "text_encoder_edit.pt"), weights_only=False)
    dn = torch.load(os.path.join(EDIT, "denoise_edit.pt"), weights_only=False)
    ve = torch.load(os.path.join(EDIT, "vae_encode.pt"), weights_only=False)
    slot_mask, img_shapes, target_hw, cond_lat = _layout(te, dn, ve)
    n_target = target_hw[0] * target_hw[1]
    assert torch.allclose(cond_lat.float(), dn["steps"][0]["hidden_states"][0, :-n_target].float(), atol=1e-2)
    dev = open_device()
    try:
        pipe = QwenImage21Pipeline(dev, load_text_encoder=False, load_vae=True, load_editing=False)
        pipe._set_target(target_hw)
        t0 = time.time()
        ps = pipe.prepare_prompt_with_images(te["prompt_embeds"][0], slot_mask, cond_lat, img_shapes[:-1], target_hw)
        print(f"\nsegments {ps.segments}\nprefix (P={ps.text_len}): {time.time()-t0:.2f}s")
        kr, vr = dn["kv_cache_step0"][31]
        kt = ttnn.to_torch(ps.kv[31][0])[0, :, : ps.text_len].permute(1, 0, 2)
        print(f"layer 31 prefix K pcc vs golden: {pcc(kt, kr[0]):.5f}")
        n = dn["num_steps"]
        lat0 = pipe.initial_latents(dn["seed"], target_hw)
        assert torch.equal(lat0, dn["steps"][0]["hidden_states"][0:1, -n_target:])
        lat = pipe.denoise_on_device(ps, lat0, n)
        t0 = time.time()
        lat = pipe.denoise_on_device(ps, lat0, n)
        dt = time.time() - t0
        p = pcc(lat, dn["latents_after_step"][n - 1][2])
        print(f"edit2 denoise {n} steps: {dt:.2f}s = {dt/n*1e3:.0f} ms/step; latents pcc vs golden = {p:.5f}")
        rgb, rgba = pipe.decode(lat)
        rgb.save(os.path.join(EDIT, "tt_edit2_golden_encoders.png"))
        print(f"pixel pcc vs reference: {_pixel_pcc(rgb, os.path.join(EDIT, 'image_edit.png')):.4f}")
        pipe.release_traces()
        assert p >= 0.99
    finally:
        close_device(dev)


def test_edit2_full_chain():
    dn = torch.load(os.path.join(EDIT, "denoise_edit.pt"), weights_only=False)
    meta = json.load(open(os.path.join(EDIT, "meta.json")))
    dev = open_device()
    try:
        pipe = QwenImage21Pipeline(dev)
        imgs = [Image.open(p) for p in (meta.get("images") or meta["image"])]
        t0 = time.time()
        rgb, rgba, lat, tm = pipe.generate(meta["prompt"], seed=meta["seed"], num_steps=meta["steps"], images=imgs)
        print(f"\nfull two-image edit chain: {json.dumps(tm.as_dict())}  wall {time.time()-t0:.1f}s")
        rgb.save(os.path.join(EDIT, "tt_edit2_full.png"))
        p = pcc(lat, dn["latents_after_step"][meta["steps"] - 1][2])
        print(
            f"latents pcc vs golden = {p:.5f}; pixel pcc vs reference = {_pixel_pcc(rgb, os.path.join(EDIT, 'image_edit.png')):.4f}"
        )
        rgb2, _, lat2, tm2 = pipe.generate(meta["prompt"], seed=meta["seed"], num_steps=meta["steps"], images=imgs)
        print(f"second run (warm): {json.dumps(tm2.as_dict())}; identical: {torch.equal(lat, lat2)}")
        pipe.release_traces()
        assert torch.equal(lat, lat2), "repeated request changed latents"
        assert p >= 0.98
    finally:
        close_device(dev)
