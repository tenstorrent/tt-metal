# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full text model PCC test — N-layer subset vs HF reference."""
import sys

import pytest
import torch
import torch.nn.functional as F

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors
from models.demos.qwen3_6_27b.tests.ttnn.test_decoder_layer_e2e import _build_hf_layer_reference, _layer_keys


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("num_layers", [4, 8])
def test_text_model_n_layer_pcc(device, num_layers):
    """N-layer text model logits PCC > 0.99 vs HF reference."""
    torch.manual_seed(123)
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"

    # Load embedding + final norm + lm_head + N decoder layers
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(keys)

    # Build TT model
    from models.demos.qwen3_6_27b.tt.model import TtQwen36Model

    tt_model = TtQwen36Model(device, weights, hf_cfg, num_layers=num_layers)

    # Build HF reference: embedding + N layers + final norm + lm head
    embed = torch.nn.Embedding(hf_cfg.vocab_size, hf_cfg.hidden_size)
    embed.weight.data = weights["model.language_model.embed_tokens.weight"].float()
    final_norm_w = weights["model.language_model.norm.weight"].float()
    lm_head_w = weights["lm_head.weight"].float()

    hf_layers = [_build_hf_layer_reference(weights, hf_cfg, i, hf_cfg.layer_types[i]).eval() for i in range(num_layers)]

    # Input
    B, T = 1, 16
    input_ids = torch.randint(0, hf_cfg.vocab_size, (B, T))

    # RoPE + mask
    rot = Qwen3NextRotaryEmbedding(hf_cfg)
    dummy = torch.zeros(B, T, hf_cfg.hidden_size)
    cos, sin = rot(dummy, torch.arange(T).unsqueeze(0))
    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    # HF gold
    with torch.no_grad():
        x = embed(input_ids).float()
        for layer in hf_layers:
            x = layer(x, cos=cos, sin=sin, attention_mask=causal_mask)
        # final norm (zero-centered, Qwen3NextRMSNorm)
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x = x_f32 * torch.rsqrt(var + hf_cfg.rms_norm_eps) * (1.0 + final_norm_w)
        hf_logits = x @ lm_head_w.T

    # TT
    tt_logits = tt_model(input_ids, cos=cos, sin=sin, attention_mask=causal_mask)

    pcc = _pcc(tt_logits, hf_logits)
    print(f"N={num_layers} layers PCC = {pcc:.6f}")

    # Top-1 token agreement
    tt_top1 = tt_logits.argmax(-1)
    hf_top1 = hf_logits.argmax(-1)
    agree = (tt_top1 == hf_top1).float().mean().item()
    print(f"  Top-1 agreement: {agree*100:.1f}%")
    assert pcc > 0.99, f"PCC {pcc:.6f} below 0.99"
