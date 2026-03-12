# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Reference: adapted from HuggingFace Qwen3.5 modeling code (Apache-2.0)
"""Pure-PyTorch reference for Qwen3.5-27B text transformer (for PCC testing)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.qwen3_5.reference.gated_delta_net import GatedDeltaNet


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_emb(q, k, cos, sin):
    """Apply RoPE only to first cos.shape[-1] dims of q, k."""
    rope_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
    cos = cos.unsqueeze(1)  # (1, 1, T, rope_dim)
    sin = sin.unsqueeze(1)
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * x * rms).to(dtype)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FullAttention(nn.Module):
    """Qwen3.5 full_attention with QK-norm, partial RoPE, and gated output."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
        # Q projection includes gate: output dim = num_heads * head_dim * 2
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        batch, seq_len, _ = x.shape
        # Q: split into query and output gate halves
        qg = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim * 2)
        query, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = gate.reshape(batch, seq_len, self.num_heads * self.head_dim)
        # QK norm (applied before RoPE)
        query = self.q_norm(query)
        key = self.k_norm(self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim))
        value = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        # Apply partial RoPE
        query = query.transpose(1, 2)  # (B, n_heads, T, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_partial_rotary_emb(query, key, cos, sin)
        # KV cache update
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            key = torch.cat([k_cache, key], dim=2)
            value = torch.cat([v_cache, value], dim=2)
        # GQA expansion
        n_rep = self.num_heads // self.num_kv_heads
        key_e = key.repeat_interleave(n_rep, dim=1)
        val_e = value.repeat_interleave(n_rep, dim=1)
        # Scaled dot-product attention with causal mask for prefill
        scale = self.head_dim**-0.5
        attn = torch.matmul(query, key_e.transpose(-2, -1)) * scale
        if seq_len > 1:
            causal = torch.triu(torch.full((seq_len, key_e.shape[2]), float("-inf"), device=x.device), diagonal=1)
            attn = attn + causal
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(attn, val_e).transpose(1, 2).reshape(batch, seq_len, -1)
        # Gated output
        out = out * torch.sigmoid(gate)
        return self.o_proj(out), (key, value)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        if self.layer_type == "full_attention":
            self.attention = FullAttention(config)
        else:
            self.attention = GatedDeltaNet(config)

    def forward(self, x, cos=None, sin=None, kv_cache=None, conv_state=None, recurrent_state=None):
        residual = x
        x = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            attn_out, new_kv = self.attention(x, cos, sin, kv_cache)
            conv_state_new, recurrent_state_new = conv_state, recurrent_state
        else:
            attn_out, conv_state_new, recurrent_state_new = self.attention(x, conv_state, recurrent_state)
            new_kv = None
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, new_kv, conv_state_new, recurrent_state_new


class Qwen3_5Config:
    """Minimal config shim built from the config.json dict."""

    def __init__(self, d: dict):
        tc = d.get("text_config", d)
        self.hidden_size = tc["hidden_size"]
        self.num_attention_heads = tc["num_attention_heads"]
        self.num_key_value_heads = tc["num_key_value_heads"]
        self.head_dim = tc.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = tc["intermediate_size"]
        self.num_hidden_layers = tc["num_hidden_layers"]
        self.vocab_size = tc["vocab_size"]
        self.rms_norm_eps = tc.get("rms_norm_eps", 1e-6)
        self.layer_types = tc["layer_types"]
        self.rope_parameters = tc.get("rope_parameters", {})
        self.linear_num_key_heads = tc.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = tc.get("linear_num_value_heads", 48)
        self.linear_key_head_dim = tc.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = tc.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = tc.get("linear_conv_kernel_dim", 4)


class Qwen3_5TextTransformer(nn.Module):
    """Qwen3.5 text-only transformer for PCC testing."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        rp = config.rope_parameters
        rope_theta = rp.get("rope_theta", 10_000_000)
        rope_dim = int(config.head_dim * rp.get("partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, position_ids: torch.Tensor):
        pos = position_ids.float()
        if pos.ndim == 3:
            pos = pos[0]  # use temporal axis for text-only
        freqs = torch.einsum("bi,j->bij", pos, self.inv_freq.to(pos.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        conv_states: Optional[list] = None,
        recurrent_states: Optional[list] = None,
        return_caches: bool = False,
    ):
        batch, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        cos, sin = self._build_cos_sin(position_ids)

        new_kv_caches, new_conv_states, new_recurrent_states = [], [], []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            cv = conv_states[i] if conv_states is not None else None
            rv = recurrent_states[i] if recurrent_states is not None else None
            x, new_kv, new_cv, new_rv = layer(x, cos, sin, kv, cv, rv)
            new_kv_caches.append(new_kv)
            new_conv_states.append(new_cv)
            new_recurrent_states.append(new_rv)

        x = self.norm(x)
        logits = self.output(x)
        if return_caches:
            return logits, new_kv_caches, new_conv_states, new_recurrent_states
        return logits
