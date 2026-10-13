# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Denoising loop (DiT only, golden text embeddings) vs the diffusers goldens, eager and traced.

    python -m pytest models/experimental/qwen_image_2_1/tests/test_pipeline.py -v -s
Env: QWEN_PIPE_STEPS (default: all 40 golden steps), QWEN_PIPE_TRACE=0 disables trace, QWEN_DIT_WDTYPE=bfp8.
"""
import os
import time

import pytest
import torch

import ttnn
from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.tt.dit import DiTPrecision
from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def dev():
    d = open_device()
    yield d
    close_device(d)


def test_denoise_matches_reference(dev):
    te = torch.load(os.path.join(GOLDENS_DIR, "text_encoder.pt"), weights_only=False)
    dn = torch.load(os.path.join(GOLDENS_DIR, "denoise.pt"), weights_only=False)
    use_trace = os.environ.get("QWEN_PIPE_TRACE", "1") == "1"
    prec = DiTPrecision()
    if os.environ.get("QWEN_DIT_WDTYPE") == "bfp8":
        prec.weight_dtype = ttnn.bfloat8_b
    fid = os.environ.get("QWEN_DIT_FIDELITY", "lofi")
    prec.norm_fp32_acc = os.environ.get("QWEN_DIT_LN_FP32", "1") == "1"
    prec.mm_fidelity = {
        "hifi2": ttnn.MathFidelity.HiFi2,
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }[fid]
    print(f"\nDiT precision: weights={prec.weight_dtype} fidelity={fid} ln_fp32_acc={prec.norm_fp32_acc}")
    have_vae = os.environ.get("QWEN_PIPE_VAE", "0") == "1"
    t0 = time.time()
    pipe = QwenImage21Pipeline(dev, dit_prec=prec, load_text_encoder=False, load_vae=have_vae, use_trace=use_trace)
    print(f"\npipeline loaded in {time.time()-t0:.1f}s (vae={have_vae}, trace={use_trace})")
    emb = te["prompt_embeds"][0]
    t0 = time.time()
    ps = pipe.prepare_prompt(emb)
    print(f"prefix: {time.time()-t0:.2f}s")
    num_steps = int(os.environ.get("QWEN_PIPE_STEPS", dn["num_steps"]))
    lat0 = pipe.initial_latents(dn["seed"])
    assert torch.equal(lat0, dn["steps"][0]["hidden_states"]), "initial noise differs from the reference"
    times = []

    def prog(i, n):
        times.append(time.time())

    loop = os.environ.get("QWEN_PIPE_LOOP", "device" if use_trace else "host")
    t0 = time.time()
    if loop == "device":
        lat = pipe.denoise_on_device(ps, lat0, num_steps)
        print(f"denoise {num_steps} steps on device (incl. trace capture): {time.time()-t0:.2f}s")
        t0 = time.time()
        lat = pipe.denoise_on_device(ps, lat0, num_steps)
        total = time.time() - t0
        print(f"denoise {num_steps} steps on device (trace replay): {total:.2f}s = {total/num_steps*1e3:.0f} ms/step")
    else:
        lat = pipe.denoise(ps, lat0, num_steps, progress=prog)
        total = time.time() - t0
        per_step = [b - a for a, b in zip(times[:-1], times[1:])]
        print(
            f"denoise {num_steps} steps: {total:.2f}s total; first step {times[0]-t0:.2f}s; median later step {sorted(per_step)[len(per_step)//2] if per_step else 0:.3f}s"
        )
    # compare with the reference latents after the same number of steps
    ref = dn["latents_after_step"][num_steps - 1][2]  # [1, 4096, 64]
    p = pcc(lat, ref)
    print(f"latents after {num_steps} steps: pcc vs reference = {p:.5f}")
    torch.save({"latents": lat, "num_steps": num_steps}, os.path.join(GOLDENS_DIR, f"tt_latents_{num_steps}.pt"))
    if have_vae and pipe.vae is not None:
        t0 = time.time()
        rgb, rgba = pipe.decode(lat)
        print(f"vae decode: {time.time()-t0:.2f}s")
        rgb.save(os.path.join(GOLDENS_DIR, f"tt_image_{num_steps}.png"))
        rgba.save(os.path.join(GOLDENS_DIR, f"tt_image_{num_steps}_rgba.png"))
    if use_trace:
        pipe.release_traces()
    assert p >= 0.99
