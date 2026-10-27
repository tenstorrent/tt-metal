# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full chain on device (text encoder -> DiT prefix -> 40-step traced loop) vs the diffusers golden latents.
    python -m pytest models/experimental/qwen_image_2_1/tests/test_e2e_latents.py -v -s
Env: QWEN_DIT_FIDELITY=lofi|hifi2 (default lofi), QWEN_PIPE_VAE=1 to also decode and save the PNG."""
import os
import time

import torch

import ttnn
from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR, PROMPT_DEMO
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.tt.dit import DiTPrecision
from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline
from models.experimental.qwen_image_2_1.tt.text_encoder import TEPrecision


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_e2e_latents():
    te = torch.load(os.path.join(GOLDENS_DIR, "text_encoder.pt"), weights_only=False)
    dn = torch.load(os.path.join(GOLDENS_DIR, "denoise.pt"), weights_only=False)
    dev = open_device()
    try:
        prec = DiTPrecision()
        fid = os.environ.get("QWEN_DIT_FIDELITY", "lofi")
        prec.mm_fidelity = {"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[fid]
        have_vae = os.environ.get("QWEN_PIPE_VAE", "0") == "1"
        t0 = time.time()
        pipe = QwenImage21Pipeline(dev, dit_prec=prec, te_prec=TEPrecision(), load_vae=have_vae)
        print(
            f"\nloaded TE+DiT{'+VAE' if have_vae else ''} in {time.time()-t0:.1f}s (dit {pipe.load_dit_s:.1f}s, te {pipe.load_te_s:.1f}s)"
        )
        prompt = te["prompt"]
        assert prompt == PROMPT_DEMO
        t0 = time.time()
        emb = pipe.encode_prompt(prompt)
        ttnn.synchronize_device(dev)
        print(
            f"text encode: {time.time()-t0:.3f}s; prompt_embeds pcc vs golden = {pcc(emb, te['prompt_embeds'][0]):.5f}"
        )
        t0 = time.time()
        ps = pipe.prepare_prompt(emb)
        print(f"prefix: {time.time()-t0:.3f}s")
        n = dn["num_steps"]
        lat0 = pipe.initial_latents(dn["seed"])
        lat = pipe.denoise_on_device(ps, lat0, n)  # capture
        t0 = time.time()
        lat = pipe.denoise_on_device(ps, lat0, n)
        dt = time.time() - t0
        ref = dn["latents_after_step"][n - 1][2]
        p = pcc(lat, ref)
        print(f"denoise {n} steps (trace replay): {dt:.2f}s = {dt/n*1e3:.0f} ms/step; latents pcc vs golden = {p:.5f}")
        torch.save({"latents": lat}, os.path.join(GOLDENS_DIR, f"tt_e2e_latents_{fid}.pt"))
        if have_vae:
            t0 = time.time()
            rgb, rgba = pipe.decode(lat)
            print(f"vae decode: {time.time()-t0:.2f}s")
            rgb.save(os.path.join(GOLDENS_DIR, f"tt_e2e_{fid}.png"))
            rgba.save(os.path.join(GOLDENS_DIR, f"tt_e2e_{fid}_rgba.png"))
        pipe.release_traces()
        assert p >= 0.99
    finally:
        close_device(dev)
