# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Consolidated PyTorch reference for Qwen3.6-27B (galaxy, text-only).

This module is the single CPU oracle for all (8,4) mesh PCC tests.
It is self-contained: imports only stdlib + torch + the validated
delta_rule_ops kernels.

Convention notes
----------------
* RMSNorm(zero_centered=True)  : output = (1 + w) * norm(x)   [HF Qwen3NextRMSNorm]
* RMSNorm(zero_centered=False) : output = w * norm(x)          [standard; used for DeltaNet inner norm]
* MRoPE section [11,11,10] partitions rotary_dim=64 into t/h/w channel groups.
  For text-only forward all three position axes are identical (token index),
  which collapses to standard RoPE — numerically matches HF Qwen3NextRotaryEmbedding.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

# ---------------------------------------------------------------------------
# Config shim
# ---------------------------------------------------------------------------


class Qwen36Config:
    """Minimal config shim built from the config.json dict (or the text_config sub-dict).

    Accepts the top-level config.json dict that may contain a ``text_config`` key;
    falls through to the dict itself if not present.
    """

    def __init__(self, d: dict):
        tc = d.get("text_config", d)
        self.hidden_size: int = tc["hidden_size"]
        self.num_attention_heads: int = tc["num_attention_heads"]
        self.num_key_value_heads: int = tc["num_key_value_heads"]
        self.head_dim: int = tc.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size: int = tc["intermediate_size"]
        self.num_hidden_layers: int = tc["num_hidden_layers"]
        self.vocab_size: int = tc["vocab_size"]
        self.rms_norm_eps: float = tc.get("rms_norm_eps", 1e-6)
        self.layer_types: list = tc["layer_types"]

        # RoPE / MRoPE
        rp = tc.get("rope_parameters", {})
        self.rope_theta: float = rp.get("rope_theta", 10_000_000)
        self.partial_rotary_factor: float = rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25))
        self.mrope_section: list = rp.get("mrope_section", [11, 11, 10])

        # Linear attention (GatedDeltaNet) dims
        self.linear_num_key_heads: int = tc.get("linear_num_key_heads", 16)
        self.linear_num_value_heads: int = tc.get("linear_num_value_heads", 48)
        self.linear_key_head_dim: int = tc.get("linear_key_head_dim", 128)
        self.linear_value_head_dim: int = tc.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim: int = tc.get("linear_conv_kernel_dim", 4)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMS normalisation with optional zero-centred weight convention.

    zero_centered=True  (default for transformer norms in Qwen3):
        output = (1 + weight) * x * rsqrt(var + eps)
        Weight is initialised to zero so that at init the layer is identity.
        Matches HF ``Qwen3NextRMSNorm``.

    zero_centered=False (used for GatedDeltaNet inner output norm):
        output = weight * x * rsqrt(var + eps)
        Weight is initialised to ones.  Matches standard LayerNorm / RMSNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = True):
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        if zero_centered:
            # HF inits to zeros so that (1 + 0) * x = x at initialisation
            self.weight = nn.Parameter(torch.zeros(dim))
        else:
            # Standard: weight=1 is identity at init
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float())
        if self.zero_centered:
            out = out * (1.0 + self.weight.float())
        else:
            out = out * self.weight.float()
        return out.type_as(x)


# ---------------------------------------------------------------------------
# MRoPE builder
# ---------------------------------------------------------------------------


def build_mrope_cos_sin(
    positions_3d: torch.Tensor,
    head_dim: int,
    partial_rotary_factor: float,
    mrope_section: list,
    theta: float,
) -> tuple:
    """Build (cos, sin) tables for MRoPE / partial RoPE.

    Args:
        positions_3d: [3, T] integer tensor — (temporal, height, width) position
            indices per token.  For text-only, set all three rows equal to
            ``torch.arange(T)``.
        head_dim:             Full head dimension (e.g. 256).
        partial_rotary_factor: Fraction of head_dim that is rotated (e.g. 0.25).
        mrope_section:        Channel groups for each position axis, e.g. [11,11,10].
            Sum must equal ``rotary_dim // 2``.
        theta:                RoPE base frequency (e.g. 10_000_000).

    Returns:
        (cos, sin) both of shape [1, T, rotary_dim].
        The returned tables should be passed directly to :class:`GatedAttention`.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)  # e.g. 64
    half_dim = rotary_dim // 2  # e.g. 32

    # Validate section sum
    assert (
        sum(mrope_section) == half_dim
    ), f"mrope_section {mrope_section} sums to {sum(mrope_section)}, expected {half_dim}"

    # Build inv_freq: 1 / theta^(2i/rotary_dim) for i in [0, half_dim)
    # shape [half_dim]
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

    # positions_3d: [3, T]
    T = positions_3d.shape[1]
    pos = positions_3d.float()  # [3, T]

    # For each of the 3 position axes, compute freqs for its channel slice
    # freq[axis, slice_i, t] = pos[axis, t] * inv_freq[slice_i]
    # Result shape: [half_dim, T]
    freqs = torch.zeros(half_dim, T, dtype=torch.float32)
    start = 0
    for axis, size in enumerate(mrope_section):
        inv_slice = inv_freq[start : start + size]  # [size]
        # outer product: [size, T]
        freqs[start : start + size] = torch.outer(inv_slice, pos[axis])
        start += size

    # freqs: [half_dim, T] -> [T, half_dim]
    freqs = freqs.T  # [T, half_dim]

    # Concatenate freqs with itself to get [T, rotary_dim] then take cos/sin
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, rotary_dim]
    cos = emb.cos().unsqueeze(0)  # [1, T, rotary_dim]
    sin = emb.sin().unsqueeze(0)  # [1, T, rotary_dim]
    return cos, sin


# ---------------------------------------------------------------------------
# Rotate half helper
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# GatedAttention (Qwen3NextAttention)
# ---------------------------------------------------------------------------


class GatedAttention(nn.Module):
    """Full-attention block matching HF ``Qwen3NextAttention`` semantics.

    Design notes
    ------------
    * ``q_proj`` output: ``n_heads * head_dim * 2`` — query and output-gate fused.
      Split via ``.view(B,T,n_q,2*head_dim).chunk(2,-1)`` → query, gate.
    * ``q_norm`` / ``k_norm``: per-head, **zero-centered** RMSNorm (matching
      HF ``Qwen3NextRMSNorm``).
    * Partial RoPE: rotate first ``rotary_dim = int(head_dim * partial_rotary_factor)``
      dims, pass-through the rest.
    * GQA expansion via ``repeat_interleave``.
    * SDPA with causal mask for prefill.
    * Output gate: ``attn_out * sigmoid(gate)``.
    * ``cos``/``sin`` are accepted from the caller — not recomputed internally.

    Forward signature::

        forward(x, cos, sin, kv_cache=None) -> (output, (key, value))

    where ``cos, sin`` have shape ``[1, T, rotary_dim]`` (or ``[B, T, rotary_dim]``).
    """

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)

        H = config.hidden_size
        # Q includes gate: dim = n_heads * head_dim * 2
        self.q_proj = nn.Linear(H, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, H, bias=False)

        # Per-head zero-centered RMSNorm (applied before RoPE)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, zero_centered=True)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, zero_centered=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            x:               [B, T, H]
            cos, sin:        [1, T, rotary_dim] (will be unsqueezed to [B,1,T,rd] internally)
            kv_cache:        (k_cache, v_cache) each [B, n_kv, T_past, head_dim] or None
            attention_mask:  4D additive mask [B, 1, T_q, T_kv] or None.
                             If None, no mask is applied (matching HF eager_attention_forward
                             with attention_mask=None — non-causal).  For proper causal
                             inference, pass a float mask with -inf in the upper triangle.

        Returns:
            (output [B, T, H], (key, value) [B, n_kv, T_now, head_dim])
        """
        B, T, _ = x.shape

        # --- Q: fused with gate ---
        qg = self.q_proj(x).view(B, T, self.num_heads, self.head_dim * 2)
        query, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        # gate: [B, T, n_heads, head_dim] → flatten to [B, T, n_heads * head_dim]
        gate = gate.reshape(B, T, self.num_heads * self.head_dim)

        # QK norm (applied per head BEFORE RoPE)
        query = self.q_norm(query)  # [B, T, n_heads, head_dim]
        key = self.k_norm(self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim))
        value = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Transpose to [B, heads, T, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Partial RoPE: rotate first rotary_dim dims
        # cos/sin: [1, T, rotary_dim] → unsqueeze to [1, 1, T, rotary_dim]
        cos_ = cos.unsqueeze(1)  # [B_or_1, 1, T, rd]
        sin_ = sin.unsqueeze(1)

        q_rot, q_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        k_rot, k_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        q_rot = (q_rot * cos_) + (_rotate_half(q_rot) * sin_)
        k_rot = (k_rot * cos_) + (_rotate_half(k_rot) * sin_)
        query = torch.cat([q_rot, q_pass], dim=-1)
        key = torch.cat([k_rot, k_pass], dim=-1)

        # KV cache update
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            key = torch.cat([k_cache, key], dim=2)
            value = torch.cat([v_cache, value], dim=2)

        # GQA expansion
        n_rep = self.num_heads // self.num_kv_heads
        key_e = key.repeat_interleave(n_rep, dim=1)
        val_e = value.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        T_q, T_kv = query.shape[2], key_e.shape[2]
        attn = torch.matmul(query, key_e.transpose(-2, -1)) * scale

        # Apply attention_mask if provided.  If None, no masking (matching HF
        # eager_attention_forward when attention_mask=None).
        if attention_mask is not None:
            # Slice to key length as HF does: attention_mask[:, :, :, :T_kv]
            attn = attn + attention_mask[:, :, :, :T_kv]

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(attn, val_e)  # [B, n_heads, T, head_dim]

        # Reshape and apply output gate
        out = out.transpose(1, 2).reshape(B, T_q, -1)  # [B, T, n_heads * head_dim]
        out = out * torch.sigmoid(gate[:, -T_q:])  # gate for query positions

        return self.o_proj(out), (key, value)


# ---------------------------------------------------------------------------
# RMSNormGated (for GatedDeltaNet inner output norm)
# ---------------------------------------------------------------------------


class RMSNormGated(nn.Module):
    """Standard RMSNorm followed by SiLU gating.

    Convention: weight * norm(x) * silu(gate)  — i.e. standard (not zero-centered).
    Matches HF ``Qwen3NextRMSNormGated``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.weight.float() * x
        x = x * F.silu(gate.float())
        return x.to(dtype)


# ---------------------------------------------------------------------------
# GatedDeltaNet (Qwen3NextGatedDeltaNet)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Linear attention block matching HF ``Qwen3NextGatedDeltaNet`` semantics.

    Weight layout (our Qwen3.6 split convention)
    ---------------------------------------------
    * ``in_proj_qkv`` [conv_dim, H]  where conv_dim = 2*n_k*hd_k + n_v*hd_v
      = n_k*hd_k (Q) || n_k*hd_k (K) || n_v*hd_v (V)  — BLOCK-WISE split
    * ``in_proj_z``   [n_v*hd_v, H]  — gate for output RMSNorm
    * ``in_proj_a``   [n_v, H]        — dt input
    * ``in_proj_b``   [n_v, H]        — beta input
    * ``conv1d``      Conv1d(conv_dim, conv_dim, kernel=4, groups=conv_dim)
    * ``A_log``       [n_v]
    * ``dt_bias``     [n_v]
    * ``norm.weight`` [hd_v]          — RMSNormGated weight (standard)
    * ``out_proj``    [H, n_v*hd_v]

    Forward signature::

        forward(x, conv_state=None, recurrent_state=None)
            -> (output [B,T,H], conv_state_new, recurrent_state_new)
    """

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads  # 48
        self.num_k_heads = config.linear_num_key_heads  # 16
        self.head_k_dim = config.linear_key_head_dim  # 128
        self.head_v_dim = config.linear_value_head_dim  # 128
        self.key_dim = self.head_k_dim * self.num_k_heads  # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads  # 6144
        self.conv_kernel_size = config.linear_conv_kernel_dim  # 4
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 10240

        H = config.hidden_size

        # Block-wise fused QKV projection (matches safetensors key in_proj_qkv)
        self.in_proj_qkv = nn.Linear(H, self.conv_dim, bias=False)
        # Gate for output RMSNorm
        self.in_proj_z = nn.Linear(H, self.value_dim, bias=False)
        # Beta and a (dt) projections
        self.in_proj_b = nn.Linear(H, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(H, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.A_log = nn.Parameter(torch.log(torch.ones(self.num_v_heads).uniform_(0, 16)))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Standard (not zero-centered) RMSNorm + SiLU gate
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.out_proj = nn.Linear(self.value_dim, H, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            hidden_states:    [B, T, H]
            conv_state:       [B, conv_dim, kernel-1] or None
            recurrent_state:  [B, n_v, K, V] or None

        Returns:
            (output [B, T, H], conv_state_new [B, conv_dim, kernel-1], recurrent_state_new)
        """
        B, T, _ = hidden_states.shape

        # 1. Project
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, conv_dim, T]
        z = self.in_proj_z(hidden_states)  # [B, T, value_dim]
        b = self.in_proj_b(hidden_states)  # [B, T, n_v]
        a = self.in_proj_a(hidden_states)  # [B, T, n_v]

        # 2. Causal depthwise conv1d + SiLU
        if conv_state is None:
            conv_state = torch.zeros(
                B, self.conv_dim, self.conv_kernel_size - 1, device=mixed_qkv.device, dtype=mixed_qkv.dtype
            )

        if T == 1:
            # Decode: rolling buffer
            combined = torch.cat([conv_state, mixed_qkv], dim=-1)  # [B, D, K]
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
            mixed_qkv = F.silu(out[:, :, -T:])
            conv_state_new = padded[:, :, -(self.conv_kernel_size - 1) :]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, conv_dim]

        # 3. Split Q/K/V (block-wise layout: Q[key_dim] | K[key_dim] | V[value_dim])
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.view(B, T, self.num_k_heads, self.head_k_dim)
        key = key.view(B, T, self.num_k_heads, self.head_k_dim)
        value = value.view(B, T, self.num_v_heads, self.head_v_dim)

        # 4. Beta and decay gate
        beta = b.sigmoid()  # [B, T, n_v]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [B, T, n_v]

        # 5. GQA expand q/k from n_k → n_v heads
        ratio = self.num_v_heads // self.num_k_heads  # 3
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=2)  # [B, T, n_v, hd_k]
            key = key.repeat_interleave(ratio, dim=2)

        # 6. Delta rule kernel
        # chunk for prefill (T>1), recurrent for decode (T==1)
        # IMPORTANT arg order differs between kernels:
        #   chunk:     (q, k, v, g, beta, ...)
        #   recurrent: (q, k, v, beta, g, ...)
        if T > 1:
            core_out, new_recurrent_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm=True,
            )
        else:
            core_out, new_recurrent_state = recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                beta=beta,
                g=g,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm=True,
            )

        # 7. GroupRMSNormGated: reshape for per-head norm
        # core_out: [B, T, n_v, hd_v] (from delta rule)
        # z:        [B, T, value_dim]
        core_out_flat = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_out_flat = self.norm(core_out_flat, z_flat)
        core_out = core_out_flat.reshape(B, T, self.value_dim)

        # 8. Output projection
        output = self.out_proj(core_out)
        return output, conv_state_new, new_recurrent_state


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Feed-forward SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# HybridDecoderLayer
# ---------------------------------------------------------------------------


class HybridDecoderLayer(nn.Module):
    """Hybrid transformer layer: pre-norm + (GatedAttention | GatedDeltaNet) + residual
    + post-norm + MLP + residual.

    Both norms are zero-centered (matching HF ``Qwen3NextDecoderLayer``).
    Dispatch to full_attention or linear_attention via ``config.layer_types[layer_idx]``.

    Forward signature::

        forward(x, cos, sin, attention_mask=None) -> (x, kv_cache, conv_state, recurrent_state)
    """

    def __init__(self, config: Qwen36Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        # Zero-centered norms (Qwen3 convention)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
        self.mlp = MLP(config)

        if self.layer_type == "full_attention":
            self.attention = GatedAttention(config)
        else:
            self.attention = GatedDeltaNet(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Returns: (x, kv_cache_new, conv_state_new, recurrent_state_new)
        """
        residual = x
        x = self.input_layernorm(x)

        if self.layer_type == "full_attention":
            attn_out, kv_cache_new = self.attention(x, cos, sin, kv_cache, attention_mask)
            conv_state_new = conv_state
            recurrent_state_new = recurrent_state
        else:
            attn_out, conv_state_new, recurrent_state_new = self.attention(x, conv_state, recurrent_state)
            kv_cache_new = None

        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, kv_cache_new, conv_state_new, recurrent_state_new


# ---------------------------------------------------------------------------
# Full text transformer
# ---------------------------------------------------------------------------


class Qwen36TextModel(nn.Module):
    """Qwen3.6 text-only transformer for PCC testing.

    Forward signature::

        forward(input_ids, cos, sin, attention_mask=None) -> logits [B, T, vocab_size]

    The caller is responsible for building ``cos`` / ``sin`` via
    :func:`build_mrope_cos_sin`.
    """

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HybridDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        # Final zero-centered RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        conv_states: Optional[list] = None,
        recurrent_states: Optional[list] = None,
        return_caches: bool = False,
    ) -> torch.Tensor:
        x = self.tok_embeddings(input_ids)

        new_kv_caches, new_conv_states, new_recurrent_states = [], [], []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            cv = conv_states[i] if conv_states is not None else None
            rv = recurrent_states[i] if recurrent_states is not None else None
            x, new_kv, new_cv, new_rv = layer(x, cos, sin, attention_mask, kv, cv, rv)
            new_kv_caches.append(new_kv)
            new_conv_states.append(new_cv)
            new_recurrent_states.append(new_rv)

        x = self.norm(x)
        logits = self.lm_head(x)

        if return_caches:
            return logits, new_kv_caches, new_conv_states, new_recurrent_states
        return logits


# ---------------------------------------------------------------------------
# Weight loader helper
# ---------------------------------------------------------------------------

_DEFAULT_SNAPSHOT = (
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def load_layer_weights_qwen36(
    safetensors_dir: str,
    layer_idx: int,
    layer_type: str,
) -> dict:
    """Load and return a state_dict for one transformer layer's attention block.

    The returned dict can be passed directly to ``GatedAttention.load_state_dict``
    or ``GatedDeltaNet.load_state_dict`` (strict=True).

    Args:
        safetensors_dir: path to the snapshot directory containing safetensors files
            and ``model.safetensors.index.json``.
        layer_idx: zero-based layer index.
        layer_type: ``"full_attention"`` or ``"linear_attention"``.

    Returns:
        Dict[str, torch.Tensor] with keys matching the module's state_dict.
    """
    from safetensors.torch import load_file as load_st

    base = Path(safetensors_dir)
    with open(base / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    if layer_type == "full_attention":
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        raw_keys = [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
            f"{prefix}.o_proj.weight",
            f"{prefix}.q_norm.weight",
            f"{prefix}.k_norm.weight",
        ]
        # Load from safetensors
        files_needed = sorted({weight_map[k] for k in raw_keys if k in weight_map})
        raw = {}
        for fn in files_needed:
            shard = load_st(str(base / fn))
            for k in raw_keys:
                if k in shard:
                    raw[k] = shard[k]
        # Map to GatedAttention state_dict key names
        sd = {
            "q_proj.weight": raw[f"{prefix}.q_proj.weight"].float(),
            "k_proj.weight": raw[f"{prefix}.k_proj.weight"].float(),
            "v_proj.weight": raw[f"{prefix}.v_proj.weight"].float(),
            "o_proj.weight": raw[f"{prefix}.o_proj.weight"].float(),
            "q_norm.weight": raw[f"{prefix}.q_norm.weight"].float(),
            "k_norm.weight": raw[f"{prefix}.k_norm.weight"].float(),
        }
    else:
        # linear_attention
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
        raw_keys = [
            f"{prefix}.in_proj_qkv.weight",
            f"{prefix}.in_proj_z.weight",
            f"{prefix}.in_proj_a.weight",
            f"{prefix}.in_proj_b.weight",
            f"{prefix}.conv1d.weight",
            f"{prefix}.A_log",
            f"{prefix}.dt_bias",
            f"{prefix}.norm.weight",
            f"{prefix}.out_proj.weight",
        ]
        files_needed = sorted({weight_map[k] for k in raw_keys if k in weight_map})
        raw = {}
        for fn in files_needed:
            shard = load_st(str(base / fn))
            for k in raw_keys:
                if k in shard:
                    raw[k] = shard[k]
        sd = {
            "in_proj_qkv.weight": raw[f"{prefix}.in_proj_qkv.weight"].float(),
            "in_proj_z.weight": raw[f"{prefix}.in_proj_z.weight"].float(),
            "in_proj_a.weight": raw[f"{prefix}.in_proj_a.weight"].float(),
            "in_proj_b.weight": raw[f"{prefix}.in_proj_b.weight"].float(),
            "conv1d.weight": raw[f"{prefix}.conv1d.weight"].float(),
            "A_log": raw[f"{prefix}.A_log"].float(),
            "dt_bias": raw[f"{prefix}.dt_bias"].float(),
            "norm.weight": raw[f"{prefix}.norm.weight"].float(),
            "out_proj.weight": raw[f"{prefix}.out_proj.weight"].float(),
        }
    return sd
