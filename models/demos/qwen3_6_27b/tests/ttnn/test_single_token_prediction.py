# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-token prediction on a real prompt — N layer subset.

Demonstrates the full inference pipeline (tokenize → embed → decoder → logits → argmax)
on a real text prompt. Limited to N layers for single-chip memory.
"""
import sys

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


@pytest.mark.parametrize("num_layers", [4])
def test_real_prompt_inference_n_layers(device, num_layers):
    """Real tokenized prompt through TT model — prove pipeline ends-to-end."""
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"

    # Tokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    prompt = "The capital of France is"
    input_ids = tok(prompt, return_tensors="pt").input_ids
    print(f"prompt: {prompt!r}")
    print(f"tokens: {input_ids[0].tolist()}")
    print(f"detokenized: {[tok.decode([t]) for t in input_ids[0].tolist()]}")

    # Load weights for embedding + N layers + final norm + lm_head
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(keys)

    from models.demos.qwen3_6_27b.tt.model import TtQwen36Model

    tt_model = TtQwen36Model(device, weights, hf_cfg, num_layers=num_layers)

    # RoPE + mask
    T = input_ids.shape[1]
    rot = Qwen3NextRotaryEmbedding(hf_cfg)
    dummy = torch.zeros(1, T, hf_cfg.hidden_size)
    cos, sin = rot(dummy, torch.arange(T).unsqueeze(0))
    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    # Forward
    logits = tt_model(input_ids, cos=cos, sin=sin, attention_mask=causal_mask)
    print(f"logits shape: {logits.shape}")

    # Next-token argmax at last position
    next_id = logits[0, -1].argmax().item()
    print(f"next-token id: {next_id} -> {tok.decode([next_id])!r}")

    # Sanity: shape and value range
    assert logits.shape == (1, T, hf_cfg.vocab_size)
    assert logits.abs().max().item() < 1e6, "logits exploded"
