# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_stage_fixture(path: Path, expected_stage: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        fixture = json.load(f)
    assert fixture["schema_version"] == 1
    assert fixture["approach"] == "dflash_block_diffusion"
    assert fixture["target_model"] == "moonshotai/Kimi-K2.5"
    assert fixture["draft_model"] == "z-lab/Kimi-K2.5-DFlash"
    assert fixture["stage"] == expected_stage
    return fixture


def tensor(value: list, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype)


def assert_close(actual: torch.Tensor, expected: list, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    torch.testing.assert_close(actual, tensor(expected), atol=atol, rtol=rtol)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    return weight.float() * x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)


def fake_pre_decoder_fused_stage(fixture: dict) -> dict:
    config = fixture["config"]
    inputs = fixture["inputs"]
    weights = fixture["weights"]
    base_hidden = tensor(inputs["base_hidden"])
    target_hidden = tensor(inputs["target_hidden"])
    noise_embedding = tensor(inputs["noise_embedding"])
    position_ids = torch.tensor(inputs["position_ids"], dtype=torch.long)
    base_norm_weight = tensor(weights["base_norm_weight"])
    lm_head_weight = tensor(weights["target_lm_head_weight"])
    fc_weight = tensor(weights["fc_weight"])
    hidden_norm_weight = tensor(weights["hidden_norm_weight"])

    base_logits = rms_norm(base_hidden, base_norm_weight, float(config["rms_norm_eps"])) @ lm_head_weight.T
    target_context = rms_norm(target_hidden @ fc_weight.T, hidden_norm_weight, float(config["rms_norm_eps"]))
    cos, sin = rotary_embedding(
        position_ids,
        head_dim=int(config["head_dim"]),
        rope_theta=float(config["rope_theta"]),
    )
    return {
        "base_logits": base_logits,
        "base_token_id": int(torch.argmax(base_logits, dim=-1)[0].item()),
        "target_context": target_context,
        "position_cos": cos,
        "position_sin": sin,
        "decoder_input": noise_embedding,
    }


def fake_decoder_layer_stage(fixture: dict) -> torch.Tensor:
    config = fixture["config"]
    inputs = fixture["inputs"]
    weights = fixture["weights"]
    hidden_states = tensor(inputs["hidden_states"])
    target_context = tensor(inputs["target_context"])
    cos = tensor(inputs["position_cos"])
    sin = tensor(inputs["position_sin"])
    eps = float(config["rms_norm_eps"])

    residual = hidden_states
    normed = rms_norm(hidden_states, tensor(weights["input_layernorm_weight"]), eps)
    hidden_states = residual + dflash_attention(
        normed,
        target_context,
        cos,
        sin,
        weights,
        num_heads=int(config["num_attention_heads"]),
        num_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        eps=eps,
    )
    residual = hidden_states
    normed = rms_norm(hidden_states, tensor(weights["post_attention_layernorm_weight"]), eps)
    return residual + dflash_mlp(normed, weights)


def fake_post_decoder_fused_stage(fixture: dict) -> dict:
    config = fixture["config"]
    inputs = fixture["inputs"]
    weights = fixture["weights"]
    hidden_states = tensor(inputs["hidden_states"])
    final_hidden = rms_norm(hidden_states, tensor(weights["final_norm_weight"]), float(config["rms_norm_eps"]))
    draft_logits = final_hidden[:, 1 - int(config["block_size"]) :, :] @ tensor(weights["target_lm_head_weight"]).T
    draft_token_ids = torch.argmax(draft_logits, dim=-1)
    anchor_position = int(inputs["anchor_position"])
    return {
        "final_hidden": final_hidden,
        "draft_logits": draft_logits,
        "draft_token_ids": draft_token_ids,
        "host_packet": {
            "type": "DRAFT_BLOCK_PROPOSAL",
            "anchor_position": anchor_position,
            "token_ids": draft_token_ids.tolist()[0],
            "positions": list(range(anchor_position + 1, anchor_position + int(config["block_size"]))),
        },
    }


def rotary_embedding(
    position_ids: torch.LongTensor,
    *,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def linear(x: torch.Tensor, weight: list) -> torch.Tensor:
    return x @ tensor(weight).T


def dflash_attention(
    hidden_states: torch.Tensor,
    target_context: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weights: dict,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    batch_size, query_len, _ = hidden_states.shape
    context_len = target_context.shape[1]

    q = linear(hidden_states, weights["q_proj_weight"]).view(batch_size, query_len, num_heads, head_dim)
    q = rms_norm(q, tensor(weights["q_norm_weight"]), eps).transpose(1, 2)

    k_context = linear(target_context, weights["k_proj_weight"])
    k_noise = linear(hidden_states, weights["k_proj_weight"])
    v_context = linear(target_context, weights["v_proj_weight"])
    v_noise = linear(hidden_states, weights["v_proj_weight"])
    k = torch.cat([k_context, k_noise], dim=1).view(batch_size, context_len + query_len, num_kv_heads, head_dim)
    v = torch.cat([v_context, v_noise], dim=1).view(batch_size, context_len + query_len, num_kv_heads, head_dim)
    k = rms_norm(k, tensor(weights["k_norm_weight"]), eps).transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if num_kv_heads != num_heads:
        groups = num_heads // num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    attn = attn.transpose(1, 2).contiguous().view(batch_size, query_len, num_heads * head_dim)
    return linear(attn, weights["o_proj_weight"])


def dflash_mlp(x: torch.Tensor, weights: dict) -> torch.Tensor:
    gate = linear(x, weights["gate_proj_weight"])
    up = linear(x, weights["up_proj_weight"])
    return linear(F.silu(gate) * up, weights["down_proj_weight"])
