# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.attention_full import (
    attention_forward_hybrid_bf16,
    attention_forward_reference_torch,
)


def _full_attn_config():
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        max_position_embeddings=4096,
        rope_interleave=False,
        rope_parameters={
            "type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
            "max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_attention_full_hybrid_matches_hf_eager(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention, Mistral4RotaryEmbedding

    torch.manual_seed(31)
    cfg = _full_attn_config()
    cfg._attn_implementation = "eager"
    attn = Mistral4Attention(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(x, position_ids=position_ids)

    expected = attention_forward_reference_torch(x, (cos, sin), position_ids, attn)
    out = attention_forward_hybrid_bf16(mesh_device, x, (cos, sin), position_ids, attn)

    ok, msg = comp_pcc(expected, out, pcc=0.94)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.15, atol=0.18)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_attention_full_hybrid_matches_hf_eager_with_hf_causal_mask(mesh_device, reset_seeds):
    from transformers.masking_utils import create_causal_mask
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention, Mistral4RotaryEmbedding

    torch.manual_seed(31)
    cfg = _full_attn_config()
    cfg._attn_implementation = "eager"
    attn = Mistral4Attention(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(x, position_ids=position_ids)

    causal = create_causal_mask(cfg, x, None, past_key_values=None, position_ids=position_ids)
    if causal is None or not isinstance(causal, torch.Tensor):
        pytest.skip("create_causal_mask did not return a torch.Tensor for this transformers build")

    expected = attention_forward_reference_torch(x, (cos, sin), position_ids, attn, attention_mask=causal)
    out = attention_forward_hybrid_bf16(mesh_device, x, (cos, sin), position_ids, attn, attention_mask=causal)

    ok, msg = comp_pcc(expected, out, pcc=0.94)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.15, atol=0.18)
    assert close, amsg
