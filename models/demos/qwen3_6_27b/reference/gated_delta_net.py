# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Reference: adapted from HuggingFace Qwen3.5 modeling code (Apache-2.0)
"""Pure-PyTorch reference for Qwen3.5 GatedDeltaNet linear attention layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    """Single-step recurrent update for decode (seq_len == 1 each call).

    query/key/value: (batch, seq_len, n_heads, head_dim)
    g:    (batch, seq_len, n_heads)
    beta: (batch, seq_len, n_heads)
    initial_state: (batch, n_heads, key_dim, value_dim) or None

    Returns: (output, final_state)
    """
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g)]

    batch, n_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = k_dim**-0.5
    query = query * scale

    state = (
        torch.zeros(batch, n_heads, k_dim, v_dim, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )

    outputs = torch.zeros(batch, n_heads, seq_len, v_dim, dtype=torch.float32, device=query.device)

    for t in range(seq_len):
        q_t = query[:, :, t]  # (batch, n_heads, k_dim)
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)
        beta_t = beta[:, :, t].unsqueeze(-1)  # (batch, n_heads, 1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (batch, n_heads, v_dim)
        delta = (v_t - kv_mem) * beta_t  # (batch, n_heads, v_dim)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    outputs = outputs.transpose(1, 2).to(query.dtype)  # (batch, seq_len, n_heads, v_dim)
    return outputs, state


class RMSNormGated(nn.Module):
    """RMS normalization followed by SiLU gating (Qwen3.5 DeltaNet output norm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.weight * x.to(dtype)
        x = x * F.silu(gate.float()).to(dtype)
        return x


class GatedDeltaNet(nn.Module):
    """Reference PyTorch implementation of one Qwen3.5 GatedDeltaNet layer.

    Weights expected under the prefix that the caller uses, loaded via
    state_dict['attention.*'] after map_hf_to_meta_keys_qwen3_5.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.ones(self.num_v_heads).uniform_(0, 16)))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Recurrent state (conv_state and recurrent_state) held externally or passed in
        self.conv_state: torch.Tensor | None = None
        self.recurrent_state: torch.Tensor | None = None

    def reset_cache(self):
        self.conv_state = None
        self.recurrent_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ):
        batch, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, T)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Causal depthwise conv1d - initialize state if not provided
        if conv_state is None:
            conv_state = torch.zeros(
                batch, self.conv_dim, self.conv_kernel_size - 1, device=mixed_qkv.device, dtype=mixed_qkv.dtype
            )

        if seq_len == 1:
            # Decode step: rolling buffer
            combined = torch.cat([conv_state, mixed_qkv], dim=-1)
            out = F.conv1d(
                combined,
                self.conv1d.weight.squeeze(1).unsqueeze(1),
                padding=0,
                groups=self.conv_dim,
            )
            mixed_qkv = F.silu(out[:, :, -1:])
            conv_state_new = combined[:, :, -(self.conv_kernel_size - 1) :]
        else:
            # Prefill: prepend saved state for causal context
            padded = torch.cat([conv_state, mixed_qkv], dim=-1)
            out = F.conv1d(padded, self.conv1d.weight.squeeze(1).unsqueeze(1), padding=0, groups=self.conv_dim)
            mixed_qkv = F.silu(out[:, :, -seq_len:])
            # save last kernel_size-1 for next decode
            conv_state_new = padded[:, :, -(self.conv_kernel_size - 1) :]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, T, conv_dim)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA-like expansion of Q/K to match num_v_heads
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_out, new_recurrent_state = recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state=recurrent_state
        )

        # Reshape for gated RMSNorm: merge batch*seq and heads
        core_out = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z_flat)
        core_out = core_out.reshape(batch, seq_len, self.value_dim)

        output = self.out_proj(core_out)
        return output, conv_state_new, new_recurrent_state
