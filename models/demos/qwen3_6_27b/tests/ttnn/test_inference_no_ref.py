# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Maximum-layer TT inference (no HF reference comparison, to save host RAM).

Pushes layer count up to find the single-chip limit. Proves the pipeline scales.
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


@pytest.mark.parametrize("num_layers", [64])
def test_inference_scales_no_ref(device, num_layers):
    """Pure TT inference — proves the model runs at this many layers."""
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    prompt = "The capital of France is"
    input_ids = tok(prompt, return_tensors="pt").input_ids

    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(keys)

    from models.demos.qwen3_6_27b.tt.model import TtQwen36Model

    tt_model = TtQwen36Model(device, weights, hf_cfg, num_layers=num_layers)
    # Free host weights now that TT has them (most copied to device DRAM)
    # (Note: some are still kept on host — embeddings, lm_head, mlp weights, norms)

    T = input_ids.shape[1]
    rot = Qwen3NextRotaryEmbedding(hf_cfg)
    cos, sin = rot(torch.zeros(1, T, hf_cfg.hidden_size), torch.arange(T).unsqueeze(0))
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    logits = tt_model(input_ids, cos=cos, sin=sin, attention_mask=mask)
    next_id = logits[0, -1].argmax().item()
    top5 = torch.topk(logits[0, -1], 5)
    print(f"N={num_layers}: next-token id={next_id}, token={tok.decode([next_id])!r}")
    print(f"  top-5: {[(tok.decode([i.item()]), v.item()) for i, v in zip(top5.indices, top5.values)]}")
    assert logits.shape == (1, T, hf_cfg.vocab_size)
    assert torch.isfinite(logits).all()
