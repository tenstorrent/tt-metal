# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests self-attention with TTNN acceleration."""

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.attention import (
    SelfAttention,
    SelfAttentionConfig,
    TTNNSelfAttention,
    TTNNQwen3NextGatedAttention,
)
from models.experimental.tt_symbiote.utils.device_management import set_device

PCC_THRESHOLD = 0.99


def _compute_pcc(torch_out, ttnn_out):
    t = torch_out.to(torch.float32)
    if isinstance(ttnn_out, TorchTTNNTensor):
        ttnn_out.elem = None
        n = ttnn_out.to_torch.to(torch.float32)
    else:
        n = ttnn.to_torch(ttnn_out).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([t.flatten(), n.flatten()]))[0, 1].item()
    diff = torch.abs(t - n)
    return pcc, torch.max(diff).item(), torch.mean(diff).item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_self_attention(device):
    """Test SELF Attention with TTNN acceleration."""
    config = SelfAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
    )
    model = SelfAttention(config).to(dtype=torch.bfloat16)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    inputs = TorchTTNNTensor(torch.randn((1, 5, 768), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNSelfAttention.from_torch(model)
    set_device(ttnn_model, device)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "SelfAttention")


def _make_qwen3_attention_random(hidden_size=2048, num_heads=16, num_kv_heads=2, head_dim=128):
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention

    config = Qwen3NextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        attention_bias=False,
    )
    attn = Qwen3NextAttention(config, layer_idx=0)
    torch.nn.init.normal_(attn.q_proj.weight, 0, 0.02)
    torch.nn.init.normal_(attn.k_proj.weight, 0, 0.02)
    torch.nn.init.normal_(attn.v_proj.weight, 0, 0.02)
    torch.nn.init.normal_(attn.o_proj.weight, 0, 0.02)
    return attn.to(torch.bfloat16), hidden_size


def _gated_attention_torch_sdpa_reference(attn, hidden_states, cos, sin):
    """Torch reference using F.scaled_dot_product_attention (matches TTNN SDPA)."""
    import torch.nn.functional as F
    from transformers.models.qwen3_next.modeling_qwen3_next import apply_rotary_pos_emb

    B, T, H = hidden_states.shape
    nh = attn.config.num_attention_heads
    nkv = attn.config.num_key_value_heads
    d = attn.head_dim
    scaling = d**-0.5

    qg = attn.q_proj(hidden_states)
    q_t, gate_t = torch.chunk(qg.view(B, T, nh, d * 2), 2, dim=-1)
    gate_t = gate_t.reshape(B, T, -1)
    q_t = attn.q_norm(q_t.view(B, T, nh, d)).transpose(1, 2)
    k_t = attn.k_norm(attn.k_proj(hidden_states).view(B, T, nkv, d)).transpose(1, 2)
    v_t = attn.v_proj(hidden_states).view(B, T, nkv, d).transpose(1, 2)
    q_t, k_t = apply_rotary_pos_emb(q_t, k_t, cos, sin)
    if nkv < nh:
        n_rep = nh // nkv
        k_t = k_t[:, :, None, :, :].expand(B, nkv, n_rep, T, d).reshape(B, nh, T, d)
        v_t = v_t[:, :, None, :, :].expand(B, nkv, n_rep, T, d).reshape(B, nh, T, d)
    attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scaling)
    attn_out = attn_out.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate_t)
    return F.linear(attn_out, attn.o_proj.weight)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_gated_attention_qwen3_coder_next(device):
    """Test Qwen3-Coder-Next gated attention PCC > 0.99."""
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

    attn, hidden_size = _make_qwen3_attention_random()
    attn.eval()
    torch.set_grad_enabled(False)

    rotary_emb = Qwen3NextRotaryEmbedding(config=attn.config)

    batch_size, seq_len = 1, 64
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    position_embeddings = rotary_emb(hidden_states, position_ids)
    outputs_torch = _gated_attention_torch_sdpa_reference(attn, hidden_states, *position_embeddings)

    ttnn_attn = TTNNQwen3NextGatedAttention.from_torch(attn)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    outputs_ttnn = ttnn_attn(hidden_states_tt, position_embeddings)
    pcc, max_diff, mean_diff = _compute_pcc(outputs_torch, outputs_ttnn)
    print(f"Qwen3NextGatedAttention PCC: {pcc:.6f}, Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
    assert pcc >= PCC_THRESHOLD, f"Qwen3NextGatedAttention PCC {pcc:.6f} below {PCC_THRESHOLD}"
