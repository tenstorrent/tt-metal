# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Gated Attention block end-to-end on single BH chip vs HF reference at layer-3 weights.

RED: TtGatedAttentionBlock does not exist. ImportError expected.
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention, Qwen3NextRotaryEmbedding

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_tt_gated_attention_layer3_pcc_vs_hf(device):
    """TtGatedAttention forward at layer-3 (full_attention layer) weights matches HF PCC > 0.99."""
    torch.manual_seed(42)
    cfg_dict = load_qwen36_config()
    text_cfg = cfg_dict["text_config"]
    hf_cfg = Qwen3NextConfig(**text_cfg)
    hf_cfg._attn_implementation = "eager"

    # 1. Load real layer-3 attention weights
    prefix = "model.language_model.layers.3.self_attn"
    keys = [
        f"{prefix}.q_proj.weight",
        f"{prefix}.k_proj.weight",
        f"{prefix}.v_proj.weight",
        f"{prefix}.o_proj.weight",
        f"{prefix}.q_norm.weight",
        f"{prefix}.k_norm.weight",
    ]
    weights = load_qwen36_tensors(keys)

    # 2. Build HF reference, load weights
    hf_block = Qwen3NextAttention(hf_cfg, layer_idx=3).eval()
    hf_block.load_state_dict(
        {
            "q_proj.weight": weights[f"{prefix}.q_proj.weight"].float(),
            "k_proj.weight": weights[f"{prefix}.k_proj.weight"].float(),
            "v_proj.weight": weights[f"{prefix}.v_proj.weight"].float(),
            "o_proj.weight": weights[f"{prefix}.o_proj.weight"].float(),
            "q_norm.weight": weights[f"{prefix}.q_norm.weight"].float(),
            "k_norm.weight": weights[f"{prefix}.k_norm.weight"].float(),
        }
    )

    # 3. Build our TT block
    from models.demos.qwen3_6_27b.tt.attention_v2 import TtGatedAttentionBlock

    tt_block = TtGatedAttentionBlock(device, weights, prefix, hf_cfg)

    # 4. Build input + RoPE freqs (no mask = causal)
    B, T, H = 1, 64, hf_cfg.hidden_size
    hidden = torch.randn(B, T, H, dtype=torch.float32) * 0.02

    # RoPE: use HF rotary embedding to produce cos/sin
    rotary = Qwen3NextRotaryEmbedding(hf_cfg)
    position_ids = torch.arange(T).unsqueeze(0)
    # rotary.forward takes (x, position_ids) where x is for shape/device
    cos, sin = rotary(hidden, position_ids)

    # 5. HF reference (no mask = pure causal handled by eager_attention_forward)
    causal_mask = torch.full((T, T), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    with torch.no_grad():
        hf_out, _ = hf_block(hidden, position_embeddings=(cos, sin), attention_mask=causal_mask)

    # 6. TT forward
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_out = tt_block(hidden_tt, cos, sin, causal_mask)
    tt_out_back = ttnn.to_torch(tt_out).float()

    print(f"shapes: tt={tt_out_back.shape}, hf={hf_out.shape}")
    assert tt_out_back.shape == hf_out.shape
    pcc = _pcc(tt_out_back, hf_out)
    print(f"PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} below 0.99"
