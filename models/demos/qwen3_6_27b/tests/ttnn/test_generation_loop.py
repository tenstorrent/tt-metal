# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-token generation loop (greedy decode) — proves the end-to-end inference cycle.

Currently re-runs full prefill every step (no KV cache yet). Slow but demonstrates
the pipeline works for actual generation, not just single forward.
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


def test_greedy_generate_5_tokens(device):
    """Generate 5 tokens from a real prompt using N=4 layer model (fast). Proves the loop works."""
    num_layers = 4
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    prompt = "Hello"
    input_ids = tok(prompt, return_tensors="pt").input_ids

    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(keys)

    from models.demos.qwen3_6_27b.tt.model import TtQwen36Model

    tt_model = TtQwen36Model(device, weights, hf_cfg, num_layers=num_layers)
    rot = Qwen3NextRotaryEmbedding(hf_cfg)

    print(f"\nstart prompt: {prompt!r}")
    generated = list(input_ids[0].tolist())

    for step in range(5):
        current_ids = torch.tensor([generated])
        T = current_ids.shape[1]
        cos, sin = rot(torch.zeros(1, T, hf_cfg.hidden_size), torch.arange(T).unsqueeze(0))
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

        logits = tt_model(current_ids, cos=cos, sin=sin, attention_mask=mask)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        print(
            f"step {step}: gen tok {next_id!r} -> {tok.decode([next_id])!r}; "
            f"running text: {tok.decode(generated)!r}"
        )

    final_text = tok.decode(generated)
    print(f"\nfinal generation: {final_text!r}")
    assert len(generated) == input_ids.shape[1] + 5
