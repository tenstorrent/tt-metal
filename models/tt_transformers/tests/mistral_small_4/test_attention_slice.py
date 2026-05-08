# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.attention_slice import (
    attention_kv_after_kv_b_bottleneck_bf16,
    attention_q_after_q_bottleneck_bf16,
)


def _tiny_attn_config():
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
def test_attention_q_bottleneck_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention

    torch.manual_seed(11)
    cfg = _tiny_attn_config()
    attn = Mistral4Attention(cfg, layer_idx=0).to(torch.bfloat16).eval()
    assert cfg.q_lora_rank is not None

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)

    with torch.no_grad():
        expected = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x)))

    q_a_eps = float(attn.q_a_layernorm.variance_epsilon)
    out = attention_q_after_q_bottleneck_bf16(
        mesh_device,
        x,
        attn.q_a_proj.weight,
        attn.q_a_layernorm.weight.data,
        q_a_eps,
        attn.q_b_proj.weight,
    )

    ok, msg = comp_pcc(expected, out, pcc=0.97)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.1, atol=0.12)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_attention_kv_bottleneck_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention

    torch.manual_seed(21)
    cfg = _tiny_attn_config()
    attn = Mistral4Attention(cfg, layer_idx=0).to(torch.bfloat16).eval()

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)

    with torch.no_grad():
        compressed = attn.kv_a_proj_with_mqa(x)
        k_pass, _k_rot = torch.split(
            compressed,
            [attn.kv_lora_rank, attn.qk_rope_head_dim],
            dim=-1,
        )
        expected = attn.kv_b_proj(attn.kv_a_layernorm(k_pass))

    kv_eps = float(attn.kv_a_layernorm.variance_epsilon)
    out = attention_kv_after_kv_b_bottleneck_bf16(
        mesh_device,
        x,
        attn.kv_a_proj_with_mqa.weight,
        int(attn.kv_lora_rank),
        attn.kv_a_layernorm.weight.data,
        kv_eps,
        attn.kv_b_proj.weight,
    )

    ok, msg = comp_pcc(expected, out, pcc=0.97)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.1, atol=0.12)
    assert close, amsg
