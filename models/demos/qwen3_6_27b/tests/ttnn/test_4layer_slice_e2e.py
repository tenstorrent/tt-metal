# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""4-layer hybrid slice (layers 0-3 = 3× DeltaNet + 1× full_attention) end-to-end PCC."""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors
from models.demos.qwen3_6_27b.tests.ttnn.test_decoder_layer_e2e import _build_hf_layer_reference, _layer_keys


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_4layer_hybrid_slice_pcc(device):
    """Stacked layers 0,1,2 (DeltaNet) + layer 3 (gated attn) PCC > 0.99 vs HF reference."""
    torch.manual_seed(42)
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"

    # Load all weights for layers 0-3
    all_keys = []
    for i in range(4):
        all_keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(all_keys)

    # Build HF dense reference layers 0-3
    hf_layers = [_build_hf_layer_reference(weights, hf_cfg, i, hf_cfg.layer_types[i]).eval() for i in range(4)]

    # Build TT layers
    from models.demos.qwen3_6_27b.tt.decoder import TtDecoderLayer

    tt_layers = [TtDecoderLayer(device, weights, i, hf_cfg.layer_types[i], hf_cfg) for i in range(4)]

    # Input
    B, T, H = 1, 32, hf_cfg.hidden_size
    hidden = torch.randn(B, T, H, dtype=torch.float32) * 0.02

    # RoPE for full_attention layers
    rot = Qwen3NextRotaryEmbedding(hf_cfg)
    pos = torch.arange(T).unsqueeze(0)
    cos, sin = rot(hidden, pos)
    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    # HF gold: pass through 4 layers
    x = hidden
    with torch.no_grad():
        for i, layer in enumerate(hf_layers):
            x = layer(x, cos=cos, sin=sin, attention_mask=causal_mask)
            print(f"  HF layer {i} ({hf_cfg.layer_types[i]}): abs mean = {x.abs().mean().item():.4f}")
    hf_final = x

    # TT: pass through 4 layers
    x_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    for i, layer in enumerate(tt_layers):
        x_tt = layer(x_tt, cos=cos, sin=sin, attention_mask=causal_mask)
        x_back = ttnn.to_torch(x_tt).float()
        layer_pcc = _pcc(x_back, hf_final if i == 3 else ttnn.to_torch(x_tt).float())  # PCC vs HF at last layer
        # Actually compare each layer to its own HF output — rerun ref step-by-step
    tt_back = ttnn.to_torch(x_tt).float()

    pcc = _pcc(tt_back, hf_final)
    print(f"4-layer slice PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
