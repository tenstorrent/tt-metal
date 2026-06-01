# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone PyTorch reference for Qwen3-Coder-Next Gated DeltaNet.

Provides numerically-exact decode and chunk-wise prefill implementations
that can be verified against HuggingFace transformers and used as golden
references for TT-NN kernel development.

Qwen3-Coder-Next text_config (relevant fields):
    hidden_size:           5120
    num_hidden_layers:     64
    layer_types:           3× linear_attention + 1× full_attention (repeating)
    linear_num_key_heads:  16
    linear_num_value_heads:48
    linear_key_head_dim:   128
    linear_value_head_dim: 128
    linear_conv_kernel_dim:4
    num_attention_heads:   24
    num_key_value_heads:   4
    head_dim:              256
    partial_rotary_factor: 0.25
    intermediate_size:     17408
    vocab_size:            248320
    rms_norm_eps:          1e-6
    output_gate_type:      swish
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen36Config:
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    full_attention_interval: int = 4
    # DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Standard attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    # FFN
    intermediate_size: int = 17408
    hidden_act: str = "silu"
    # Norm / misc
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Single-step (decode) or multi-step recurrent DeltaNet.

    Args:
        query:  [B, S, num_v_heads, k_dim]
        key:    [B, S, num_v_heads, k_dim]  (already expanded from num_k_heads)
        value:  [B, S, num_v_heads, v_dim]
        g:      [B, S, num_v_heads]  (raw log-decay, before exp)
        beta:   [B, S, num_v_heads]  (already sigmoid-ed)
        initial_state: [B, num_v_heads, k_dim, v_dim] or None

    Returns:
        output: [B, S, num_v_heads, v_dim]
        final_state: [B, num_v_heads, k_dim, v_dim] or None
    """
    orig_dtype = query.dtype

    if use_qk_l2norm:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    B, H, S, Dk = key.shape
    Dv = value.shape[-1]
    scale = Dk**-0.5
    query = query * scale

    output = torch.zeros(B, H, S, Dv, dtype=torch.float32, device=query.device)
    state = (
        torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.float()
    )

    for i in range(S):
        q_t = query[:, :, i]  # [B, H, Dk]
        k_t = key[:, :, i]
        v_t = value[:, :, i]  # [B, H, Dv]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        beta_t = beta[:, :, i].unsqueeze(-1)  # [B, H, 1]

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, Dv]
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = state if output_final_state else None
    output = output.transpose(1, 2).contiguous().to(orig_dtype)
    return output, final_state


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunk-wise parallel DeltaNet (prefill).

    Same shapes as recurrent_gated_delta_rule.
    """
    orig_dtype = query.dtype

    if use_qk_l2norm:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    B, H, S, Dk = key.shape
    Dv = value.shape[-1]
    pad_size = (chunk_size - S % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    S_padded = S + pad_size

    scale = Dk**-0.5
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    num_chunks = S_padded // chunk_size
    query = query.reshape(B, H, num_chunks, chunk_size, Dk)
    key = key.reshape(B, H, num_chunks, chunk_size, Dk)
    value = value.reshape(B, H, num_chunks, chunk_size, Dv)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, Dk)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, Dv)
    g = g.reshape(B, H, num_chunks, chunk_size)

    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g_cum = g.cumsum(dim=-1)
    decay_mask = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    state = (
        torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.float()
    )
    output = torch.zeros_like(value)
    mask_strict_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for c in range(num_chunks):
        q_c = query[:, :, c]
        k_c = key[:, :, c]
        v_c = value[:, :, c]

        intra_attn = q_c @ k_c.transpose(-1, -2) * decay_mask[:, :, c]
        v_prime = k_cumdecay[:, :, c] @ state
        v_new = v_c - v_prime
        attn_inter = (q_c * g_cum[:, :, c, :, None].exp()) @ state

        output[:, :, c] = attn_inter + intra_attn @ v_new
        state = (
            state * g_cum[:, :, c, -1, None, None].exp()
            + (k_c * (g_cum[:, :, c, -1, None] - g_cum[:, :, c]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    final_state = state if output_final_state else None
    output = output.reshape(B, H, S_padded, Dv)[:, :, :S]
    output = output.transpose(1, 2).contiguous().to(orig_dtype)
    return output, final_state


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(orig_dtype)
        return x * F.silu(gate.float()).to(orig_dtype)


class GatedDeltaNetLayer(nn.Module):
    """
    Full Gated DeltaNet layer matching HuggingFace Qwen3_5GatedDeltaNet.

    Conv1d → Q/K/V split → L2norm → decay/beta gates → recurrent/chunk update → gated RMSNorm → out_proj
    """

    def __init__(self, config: Qwen36Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.head_expand_ratio = self.num_v_heads // self.num_k_heads

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            hidden_states: [B, S, hidden_size]
            conv_state: [B, conv_dim, conv_kernel_size-1] or None
            recurrent_state: [B, num_v_heads, k_dim, v_dim] or None

        Returns:
            output: [B, S, hidden_size]
            new_conv_state
            new_recurrent_state
        """
        B, S, _ = hidden_states.shape
        is_decode = (S == 1) and (conv_state is not None)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, conv_dim_qkv, S]
        z = self.in_proj_z(hidden_states).reshape(B, S, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # [B, S, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [B, S, num_v_heads]

        if is_decode:
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1:] = mixed_qkv
            w = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel_size]
            mixed_qkv = (conv_state * w).sum(dim=-1, keepdim=True)
            mixed_qkv = F.silu(mixed_qkv)
        else:
            if conv_state is not None:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = conv_state[:, :, -self.conv_kernel_size:]
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
            if conv_state is not None and S < mixed_qkv.shape[-1]:
                mixed_qkv = mixed_qkv[:, :, -S:]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S, conv_dim]
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(B, S, self.num_k_heads, self.head_k_dim)
        key = key.reshape(B, S, self.num_k_heads, self.head_k_dim)
        value = value.reshape(B, S, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.head_expand_ratio > 1:
            query = query.repeat_interleave(self.head_expand_ratio, dim=2)
            key = key.repeat_interleave(self.head_expand_ratio, dim=2)

        if is_decode:
            output, recurrent_state = recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            output, recurrent_state = chunk_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(B, S, -1)
        output = self.out_proj(output)

        return output, conv_state, recurrent_state


class GatedAttentionLayer(nn.Module):
    """
    Standard GQA with output gating (sigmoid) and Q/K RMSNorm + partial RoPE.
    Matches HuggingFace Qwen3_5Attention.
    """

    def __init__(self, config: Qwen36Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, S, _ = hidden_states.shape

        q_gate = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim * 2)
        query, gate = q_gate.chunk(2, dim=-1)  # each [B, S, num_heads, head_dim]
        gate = gate.reshape(B, S, -1)  # [B, S, num_heads * head_dim]

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query, key = self._apply_partial_rotary(query, key, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            key = torch.cat([k_cache, key], dim=2)
            value = torch.cat([v_cache, value], dim=2)
        new_kv_cache = (key, value)

        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        causal_mask = torch.triu(torch.full((S, key.shape[2]), float("-inf"), device=query.device), diagonal=key.shape[2] - S + 1)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_kv_cache

    def _apply_partial_rotary(self, q, k, cos, sin):
        d = self.rotary_dim
        q_rot, q_pass = q[..., :d], q[..., d:]
        k_rot, k_pass = k[..., :d], k[..., d:]
        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MLP(nn.Module):
    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig_dtype)


class HybridDecoderLayer(nn.Module):
    def __init__(self, config: Qwen36Config, layer_idx: int):
        super().__init__()
        self.layer_type = "linear_attention" if (layer_idx % config.full_attention_interval != config.full_attention_interval - 1) else "full_attention"

        if self.layer_type == "linear_attention":
            self.token_mixer = GatedDeltaNetLayer(config, layer_idx)
        else:
            self.token_mixer = GatedAttentionLayer(config, layer_idx)

        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states, conv_state, recurrent_state = self.token_mixer(
                hidden_states,
                conv_state=kwargs.get("conv_state"),
                recurrent_state=kwargs.get("recurrent_state"),
            )
            kwargs_out = {"conv_state": conv_state, "recurrent_state": recurrent_state}
        else:
            hidden_states, kv_cache = self.token_mixer(
                hidden_states,
                cos=kwargs.get("cos"),
                sin=kwargs.get("sin"),
                kv_cache=kwargs.get("kv_cache"),
            )
            kwargs_out = {"kv_cache": kv_cache}

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kwargs_out
