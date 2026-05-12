# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Basic performance measurement — prefill TTFT and approximate tok/s.

NOTE: current implementation re-prefills every token (no KV cache), so tok/s
here is best understood as "prefill throughput per token of context."
RMSNorm/MLP/GroupRMSNorm currently run host-side; performance is unoptimized.
"""
import sys
import time

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
from transformers import AutoTokenizer
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors
from models.demos.qwen3_6_27b.tests.ttnn.test_decoder_layer_e2e import _layer_keys


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize(
    "num_layers,prefill_len",
    [
        (16, 5),  # smaller model for faster iteration
        (64, 5),  # full model
    ],
)
def test_prefill_ttft(device, num_layers, prefill_len):
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

    # Prompt with at least prefill_len tokens
    base_prompt = "The capital of France is the city of Paris which is also" * 4
    input_ids = tok(base_prompt, return_tensors="pt").input_ids[:, :prefill_len]
    T = input_ids.shape[1]
    print(f"\n=== N={num_layers} layers, prefill_len={T} tokens ===")

    # Load weights (timed separately for transparency)
    t0 = time.perf_counter()
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(keys)
    t_load = time.perf_counter() - t0
    print(f"  weight load (host): {t_load:.2f}s")

    t0 = time.perf_counter()
    from models.demos.qwen3_6_27b.tt.model import TtQwen36Model

    tt_model = TtQwen36Model(device, weights, hf_cfg, num_layers=num_layers)
    t_build = time.perf_counter() - t0
    print(f"  model build + device transfer: {t_build:.2f}s")

    rot = Qwen3NextRotaryEmbedding(hf_cfg)
    cos, sin = rot(torch.zeros(1, T, hf_cfg.hidden_size), torch.arange(T).unsqueeze(0))
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    # Warmup forward
    print("  warmup forward...")
    t0 = time.perf_counter()
    _ = tt_model(input_ids, cos=cos, sin=sin, attention_mask=mask)
    t_warmup = time.perf_counter() - t0
    print(f"  warmup time: {t_warmup:.2f}s")

    # Timed forward (3 reps, take median)
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        _ = tt_model(input_ids, cos=cos, sin=sin, attention_mask=mask)
        times.append(time.perf_counter() - t0)
    times.sort()
    ttft = times[1]  # median
    print(f"  forward times (s): {[f'{t:.2f}' for t in times]}")
    print(f"  TTFT (median): {ttft:.2f}s = {ttft*1000:.0f} ms")
    print(f"  tok/s for 1-token decode (full re-prefill): {1/ttft:.3f}")
    print(f"  prefill throughput: {T/ttft:.2f} tok/s")
