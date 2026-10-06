# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the TTNN DiT against the diffusers goldens.

    python -m pytest models/experimental/qwen_image_2_1/tests/test_dit.py -v -s
Env: QWEN_DIT_LAYERS=<n> restricts the number of blocks (layer-wise debugging), QWEN_DIT_WDTYPE=bfp8.
"""
import os
import time

import pytest
import torch

import ttnn
from models.experimental.qwen_image_2_1.common import rope as rope_mod
from models.experimental.qwen_image_2_1.common import schedule
from models.experimental.qwen_image_2_1.common.config import DIT, GOLDENS_DIR
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.common.weights import transformer_ckpt
from models.experimental.qwen_image_2_1.tt.dit import DeviceCond, DiTPrecision, QwenImageDiT

N_LAYERS = int(os.environ.get("QWEN_DIT_LAYERS", "32"))
H = W = 64  # 1024x1024 -> 64x64 latent tokens


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def dev():
    d = open_device(trace_region_size=64 * 1024 * 1024)
    yield d
    close_device(d)


@pytest.fixture(scope="module")
def goldens():
    te = torch.load(os.path.join(GOLDENS_DIR, "text_encoder.pt"), weights_only=False)
    dn = torch.load(os.path.join(GOLDENS_DIR, "denoise.pt"), weights_only=False)
    return te, dn


@pytest.fixture(scope="module")
def model(dev):
    ck = transformer_ckpt()
    prec = DiTPrecision()
    if os.environ.get("QWEN_DIT_WDTYPE") == "bfp8":
        prec.weight_dtype = ttnn.bfloat8_b
    t0 = time.time()
    m = QwenImageDiT(dev, ck, prec, layers=N_LAYERS)
    print(f"\nloaded {N_LAYERS} layers to device in {time.time()-t0:.1f}s")
    return m, ck


def _conds(model, t01):
    tc = model.time_cond
    sc = schedule.modulation_rows_bf16_like_reference(tc, t01)
    return sc, DeviceCond.from_host(model.dev, sc)


def test_zero_layers_does_not_load_transformer_blocks(dev):
    checkpoint = transformer_ckpt()
    try:
        shell = QwenImageDiT(dev, checkpoint, layers=0)
        assert shell.layers == []
    finally:
        checkpoint.close()


def test_prefix_kv_matches_reference(dev, model, goldens):
    m, ck = model
    te, dn = goldens
    text = te["prompt_embeds"][0]  # [T, 4096] bf16
    T = text.shape[0]
    cos, sin = rope_mod.dit_cos_sin(T, H, W)
    rt_text = m.rope_tables(cos[:T], sin[:T])
    sc0, cond0 = _conds(m, 0.0)
    t0 = time.time()
    kv, T2 = m.prefix_kv(text, rt_text, cond0)
    ttnn.synchronize_device(dev)
    print(f"prefix_kv: {time.time()-t0:.2f}s (eager, incl. compile)")
    assert T2 == T
    worst = 1.0
    for li, (k, v) in enumerate(kv):
        kr, vr = dn["kv_cache_step0"][li]  # [1, T, 32, 128]
        # the plain (streaming) t2i attention keeps the tile-padded rows in K/V; compare the real T rows only
        kt = ttnn.to_torch(k)[0, :, :T].permute(1, 0, 2)  # [T, 32, 128]
        vt = ttnn.to_torch(v)[0, :, :T].permute(1, 0, 2)
        pk, pv = pcc(kt, kr[0]), pcc(vt, vr[0])
        worst = min(worst, pk, pv)
        if li < 3 or li % 8 == 7 or min(pk, pv) < 0.99:
            print(f"layer {li:2d}: pcc k={pk:.5f} v={pv:.5f}")
    print(f"worst kv pcc over {len(kv)} layers: {worst:.5f}")
    assert worst > 0.98


def test_block0_image_rows_vs_reference(dev, model, goldens):
    """Block-0 output of the joint step-0 pass (image rows) vs the diffusers tap."""
    m, ck = model
    te, dn = goldens
    if 0 not in dn.get("block_taps", {}) or 0 not in dn["block_taps"][0]:
        pytest.skip("no block taps in goldens")
    text = te["prompt_embeds"][0]
    T = text.shape[0]
    cos, sin = rope_mod.dit_cos_sin(T, H, W)
    rt_text = m.rope_tables(cos[:T], sin[:T])
    rt_img = m.rope_tables(cos[T:], sin[T:])
    _, cond0 = _conds(m, 0.0)
    kv, _ = m.prefix_kv(text, rt_text, cond0)
    lat = dn["steps"][0]["hidden_states"]  # [1, 4096, 64] bf16
    t01 = float(dn["steps"][0]["timestep"][0])
    _, cond = _conds(m, t01)
    x = ttnn.from_torch(
        lat.reshape(1, 1, -1, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out, taps = m.step(x, cond, kv, rt_img, T, return_hidden_taps=[0])
    ref_tap = dn["block_taps"][0][0][1][0, T:]  # [4096, 4096] image rows after block 0
    p = pcc(taps[0].reshape(-1, ref_tap.shape[-1]), ref_tap)
    print(f"block0 image-rows pcc = {p:.5f}")
    assert p > 0.99


@pytest.mark.parametrize("step_idx", [1, 0])
def test_step_matches_reference(dev, model, goldens, step_idx):
    m, ck = model
    te, dn = goldens
    if N_LAYERS != DIT.num_layers:
        pytest.skip("full model only")
    text = te["prompt_embeds"][0]
    T = text.shape[0]
    cos, sin = rope_mod.dit_cos_sin(T, H, W)
    rt_text = m.rope_tables(cos[:T], sin[:T])
    rt_img = m.rope_tables(cos[T:], sin[T:])
    _, cond0 = _conds(m, 0.0)
    kv, _ = m.prefix_kv(text, rt_text, cond0)
    st = dn["steps"][step_idx]
    lat = st["hidden_states"]
    t01 = float(st["timestep"][0])
    _, cond = _conds(m, t01)
    x = ttnn.from_torch(
        lat.reshape(1, 1, -1, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    t0 = time.time()
    out = m.step(x, cond, kv, rt_img, T)
    ttnn.synchronize_device(dev)
    print(f"step (eager): {time.time()-t0:.2f}s")
    t0 = time.time()
    out = m.step(x, cond, kv, rt_img, T)
    ttnn.synchronize_device(dev)
    print(f"step (eager, warm): {time.time()-t0:.2f}s")
    o = ttnn.to_torch(out).reshape(-1, 64)  # [4096, 64]
    ref = st["noise_pred"][0]
    if ref.shape[0] != o.shape[0]:
        ref = ref[-o.shape[0] :]
    p = pcc(o, ref)
    print(f"step {step_idx} (t={t01:.4f}) noise_pred pcc = {p:.5f}  max|d|={(o.float()-ref.float()).abs().max():.4f}")
    assert p > 0.98
