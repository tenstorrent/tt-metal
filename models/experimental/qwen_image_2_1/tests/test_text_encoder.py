# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-8B text encoder (TTNN) vs the transformers goldens (per-layer hidden states).

    python -m pytest models/experimental/qwen_image_2_1/tests/test_text_encoder.py -v -s
Env: QWEN_TE_LAYERS=<n> (default 36), QWEN_TE_WDTYPE=bf16|bfp8 (default bfp8).
"""
import os
import time

import pytest
import torch

import ttnn
from models.experimental.qwen_image_2_1.common import text as text_mod
from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR, TE
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.common.weights import text_encoder_ckpt
from models.experimental.qwen_image_2_1.tt.text_encoder import Qwen3VLTextEncoder, TEPrecision

N_LAYERS = int(os.environ.get("QWEN_TE_LAYERS", str(TE.num_layers)))


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def dev():
    d = open_device(trace_region_size=64 * 1024 * 1024)
    yield d
    close_device(d)


def test_text_encoder_matches_reference(dev):
    te = torch.load(os.path.join(GOLDENS_DIR, "text_encoder.pt"), weights_only=False)
    hs = torch.load(os.path.join(GOLDENS_DIR, "text_encoder_hidden_states.pt"), weights_only=False)["hidden_states"]
    ids = te["input_ids"]
    assert text_mod.tokenize_prompt(te["prompt"]).tolist() == ids.tolist()
    prec = TEPrecision()
    if os.environ.get("QWEN_TE_WDTYPE", "bfp8") == "bf16":
        prec.weight_dtype = ttnn.bfloat16
    ck = text_encoder_ckpt()
    t0 = time.time()
    m = Qwen3VLTextEncoder(dev, ck, prec, layers=N_LAYERS)
    print(f"\nloaded {N_LAYERS} TE layers ({prec.weight_dtype}) in {time.time()-t0:.1f}s")
    emb = m.embed(ids)
    assert pcc(emb, hs[0][0]) > 0.9999, "embedding gather mismatch"
    taps = list(range(N_LAYERS))
    t0 = time.time()
    out, tp = m.encode(ids, taps=taps)
    ttnn.synchronize_device(dev)
    print(f"encode (eager, with taps): {time.time()-t0:.2f}s")
    worst = 1.0
    for li in taps:
        p = pcc(tp[li], hs[li + 1][0])
        worst = min(worst, p)
        if li < 4 or li % 6 == 5 or p < 0.99:
            print(f"layer {li:2d} out pcc = {p:.5f}")
    print(f"worst layer pcc = {worst:.5f}")
    if N_LAYERS == TE.num_layers:
        p_final = pcc(out, hs[-1][0])
        p_dropped = pcc(out[text_mod.DROP_IDX :], te["prompt_embeds"][0])
        print(f"final hidden pcc = {p_final:.5f}; prompt_embeds (after drop) pcc = {p_dropped:.5f}")
        assert p_dropped > 0.99
    # timing without taps
    t0 = time.time()
    out2 = m.encode(ids)
    ttnn.synchronize_device(dev)
    print(f"encode (eager, warm, no taps): {time.time()-t0:.3f}s")
    assert worst > 0.98
