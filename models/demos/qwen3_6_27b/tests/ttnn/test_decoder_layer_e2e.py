# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single hybrid decoder layer end-to-end vs HF Qwen3NextDecoderLayer.

Tests both layer types: layer 0 (linear_attention) and layer 3 (full_attention).
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextRMSNorm,
    Qwen3NextRotaryEmbedding,
)

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors


def _build_hf_layer_reference(weights, hf_cfg, layer_idx, layer_type):
    """Build a dense (non-MoE) layer that mirrors HF Qwen3NextDecoderLayer semantics."""
    import torch.nn as nn

    base = f"model.language_model.layers.{layer_idx}"

    class DenseLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps)
            self.post_attention_layernorm = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps)
            self.gate_proj = nn.Linear(hf_cfg.hidden_size, hf_cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(hf_cfg.hidden_size, hf_cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(hf_cfg.intermediate_size, hf_cfg.hidden_size, bias=False)
            if layer_type == "linear_attention":
                self.attn = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=layer_idx)
            else:
                self.attn = Qwen3NextAttention(hf_cfg, layer_idx=layer_idx)
            self.layer_type = layer_type

        def forward(self, x, cos=None, sin=None, attention_mask=None):
            r = x
            x = self.input_layernorm(x)
            if self.layer_type == "linear_attention":
                out = self.attn(x, cache_position=torch.arange(x.shape[1]))
                if isinstance(out, tuple):
                    out = out[0]
            else:
                out, _ = self.attn(x, position_embeddings=(cos, sin), attention_mask=attention_mask)
            x = r + out
            r = x
            x = self.post_attention_layernorm(x)
            mlp = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
            return r + mlp

    layer = DenseLayer().eval()

    sd = {
        "input_layernorm.weight": weights[f"{base}.input_layernorm.weight"].float(),
        "post_attention_layernorm.weight": weights[f"{base}.post_attention_layernorm.weight"].float(),
        "gate_proj.weight": weights[f"{base}.mlp.gate_proj.weight"].float(),
        "up_proj.weight": weights[f"{base}.mlp.up_proj.weight"].float(),
        "down_proj.weight": weights[f"{base}.mlp.down_proj.weight"].float(),
    }
    if layer_type == "linear_attention":
        from models.demos.qwen3_6_27b.tests.ttnn.test_deltanet_block_e2e import _reconstruct_hf_qkvz_ba

        prefix = f"{base}.linear_attn"
        qkvz, ba = _reconstruct_hf_qkvz_ba(
            prefix,
            weights,
            n_v=hf_cfg.linear_num_value_heads,
            n_k=hf_cfg.linear_num_key_heads,
            hd_k=hf_cfg.linear_key_head_dim,
            hd_v=hf_cfg.linear_value_head_dim,
        )
        sd.update(
            {
                "attn.in_proj_qkvz.weight": qkvz,
                "attn.in_proj_ba.weight": ba,
                "attn.conv1d.weight": weights[f"{prefix}.conv1d.weight"].float(),
                "attn.A_log": weights[f"{prefix}.A_log"].float(),
                "attn.dt_bias": weights[f"{prefix}.dt_bias"].float(),
                "attn.norm.weight": weights[f"{prefix}.norm.weight"].float(),
                "attn.out_proj.weight": weights[f"{prefix}.out_proj.weight"].float(),
            }
        )
    else:
        prefix = f"{base}.self_attn"
        for k in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight"]:
            sd[f"attn.{k}"] = weights[f"{prefix}.{k}"].float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert len(missing) == 0, f"reference layer missing: {missing}"
    return layer


import torch.nn.functional as F


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _layer_keys(layer_idx, layer_type):
    """All weight keys for one decoder layer."""
    base = f"model.language_model.layers.{layer_idx}"
    common = [
        f"{base}.input_layernorm.weight",
        f"{base}.post_attention_layernorm.weight",
        f"{base}.mlp.gate_proj.weight",
        f"{base}.mlp.up_proj.weight",
        f"{base}.mlp.down_proj.weight",
    ]
    if layer_type == "linear_attention":
        return common + [
            f"{base}.linear_attn.in_proj_qkv.weight",
            f"{base}.linear_attn.in_proj_a.weight",
            f"{base}.linear_attn.in_proj_b.weight",
            f"{base}.linear_attn.in_proj_z.weight",
            f"{base}.linear_attn.conv1d.weight",
            f"{base}.linear_attn.A_log",
            f"{base}.linear_attn.dt_bias",
            f"{base}.linear_attn.norm.weight",
            f"{base}.linear_attn.out_proj.weight",
        ]
    else:  # full_attention
        return common + [
            f"{base}.self_attn.q_proj.weight",
            f"{base}.self_attn.k_proj.weight",
            f"{base}.self_attn.v_proj.weight",
            f"{base}.self_attn.o_proj.weight",
            f"{base}.self_attn.q_norm.weight",
            f"{base}.self_attn.k_norm.weight",
        ]


@pytest.mark.parametrize("layer_idx,layer_type", [(0, "linear_attention"), (3, "full_attention")])
def test_tt_decoder_layer_pcc_vs_hf(device, layer_idx, layer_type):
    """TtDecoderLayer forward matches HF Qwen3NextDecoderLayer PCC > 0.99."""
    torch.manual_seed(42)
    cfg_dict = load_qwen36_config()
    text_cfg = cfg_dict["text_config"]
    hf_cfg = Qwen3NextConfig(**text_cfg)
    hf_cfg._attn_implementation = "eager"

    keys = _layer_keys(layer_idx, layer_type)
    weights = load_qwen36_tensors(keys)

    # Build HF-equivalent dense layer (DeltaNet/Attention from HF + dense MLP + HF norms)
    ref_layer = _build_hf_layer_reference(weights, hf_cfg, layer_idx, layer_type)

    # Build our TT decoder layer
    from models.demos.qwen3_6_27b.tt.decoder import TtDecoderLayer

    tt_layer = TtDecoderLayer(device, weights, layer_idx, layer_type, hf_cfg)

    # Input
    B, T, H = 1, 32, hf_cfg.hidden_size
    hidden = torch.randn(B, T, H, dtype=torch.float32) * 0.02

    # RoPE for full_attention layer
    rot = Qwen3NextRotaryEmbedding(hf_cfg)
    pos = torch.arange(T).unsqueeze(0)
    cos, sin = rot(hidden, pos)
    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ref_out = ref_layer(hidden, cos=cos, sin=sin, attention_mask=causal_mask)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        print(f"  ref_out abs mean: {ref_out.abs().mean().item():.6f}")
    hf_out = ref_out

    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_out = tt_layer(hidden_tt, cos=cos, sin=sin, attention_mask=causal_mask)
    tt_out_back = ttnn.to_torch(tt_out).float()

    assert tt_out_back.shape == hf_out.shape
    pcc = _pcc(tt_out_back, hf_out)
    print(f"Layer {layer_idx} ({layer_type}): PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} below 0.99"
