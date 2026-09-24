# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone torch reference for Qwen3.8-27B (text model, prefill only).

Pure torch — no ttnn, no device code. Trimmed and vendored from upstream
``transformers/models/qwen3_5/modeling_qwen3_5.py`` (transformers 5.12.1); provenance per block:

  Qwen3_5TextRotaryEmbedding      L95-184   (interleaved M-RoPE; text-only positions => 1D RoPE)
  Qwen3_5RMSNormGated             L187-202
  l2norm                          L240-243
  torch_chunk_gated_delta_rule    L246-324
  Qwen3_5GatedDeltaNet.forward    L437-559  (cache-free prefill + explicit conv/recurrent carry)
  rotate_half / apply_rotary      L562-605
  eager_attention_forward         L620-642
  Qwen3_5Attention.forward        L672-717
  Qwen3_5MLP                      L720-733
  Qwen3_5RMSNorm                  L736-753  (zero-centred: x * (1 + w))
  Qwen3_5DecoderLayer             L756-809
  Qwen3_5TextModel.forward        L1155-1218

Differences from upstream, all deliberate:
  * State is explicit: every token mixer takes and returns its carried state (GDN: conv_state
    ``[B, conv_dim, K-1]`` = the last K-1 *pre-conv* projection columns, recurrent_state
    ``[B, Nv, Dk, Dv]`` fp32; attention: post-RoPE K and raw V ``[B, Hkv, T, D]``), so a chunked
    run can be compared against a one-shot run without a transformers Cache object.
  * Compute dtype follows the inputs (bf16 per the recipe's reference convention). Upstream's own
    internal fp32 upcasts (norms, the delta-rule core, softmax) are kept.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.demos.qwen_3_8_27b.config import Qwen38Config


# ------------------------------------------------------------------------------------------------
# norms
# ------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Zero-centred RMSNorm: ``norm(x) * (1 + w)``, computed in fp32 then cast back."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class RMSNormGated(nn.Module):
    """GDN output norm: plain-weight RMSNorm (no +1), then ``* silu(gate)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dt)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(dt)


# ------------------------------------------------------------------------------------------------
# RoPE
# ------------------------------------------------------------------------------------------------
def rope_inv_freq(cfg: Qwen38Config) -> torch.Tensor:
    dim = cfg.rotary_dim
    return 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def rope_cos_sin(cfg: Qwen38Config, positions: torch.Tensor, dtype=torch.bfloat16):
    """cos/sin ``[T, rotary_dim]`` for 1D text positions.

    Upstream builds 3 position streams (T, H, W) and interleaves them per ``mrope_section``; for
    text-only input all three streams equal the token position, so the interleave is the identity
    and this reduces to plain (neox half-split) RoPE over the first ``rotary_dim`` channels.
    ``test_config_and_reference.py`` checks this equivalence against upstream's rotary module.
    """
    inv = rope_inv_freq(cfg)
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(x, cos, sin):
    """x: ``[B, H, T, D]``; cos/sin ``[T, rotary_dim]``. Rotates the first rotary_dim channels."""
    rd = cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


# ------------------------------------------------------------------------------------------------
# Gated DeltaNet
# ------------------------------------------------------------------------------------------------
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=None):
    """Upstream ``torch_chunk_gated_delta_rule`` with qk-l2norm in kernel and the final state out.

    q, k: ``[B, T, H, Dk]`` (already expanded to Nv heads); v ``[B, T, H, Dv]``; g, beta ``[B, T, H]``.
    Returns ``o [B, T, H, Dv]`` (input dtype) and ``state [B, H, Dk, Dv]`` fp32.
    """
    initial_dtype = q.dtype
    q = l2norm(q)
    k = l2norm(k)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    q = F.pad(q, (0, 0, 0, pad))
    k = F.pad(k, (0, 0, 0, pad))
    v = F.pad(v, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    L = T + pad
    q = q * (1 / (Dk**0.5))
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)]
    g = g.reshape(B, H, -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32) if initial_state is None else initial_state.float()
    out = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(L // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ S + a @ v_new
        S = (
            S * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(initial_dtype)
    return out, S


def recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None):
    """Token-by-token delta rule (upstream ``torch_recurrent_gated_delta_rule``) — the slow oracle."""
    initial_dtype = q.dtype
    q = l2norm(q)
    k = l2norm(k)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]
    B, H, T, Dk = k.shape
    q = q * (1 / (Dk**0.5))
    S = torch.zeros(B, H, Dk, v.shape[-1]) if initial_state is None else initial_state.float().clone()
    out = torch.zeros(B, H, T, v.shape[-1])
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kv_mem = (S * k[:, :, t].unsqueeze(-1)).sum(-2)
        delta = (v[:, :, t] - kv_mem) * beta[:, :, t].unsqueeze(-1)
        S = S + k[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (S * q[:, :, t].unsqueeze(-1)).sum(-2)
    return out.transpose(1, 2).contiguous().to(initial_dtype), S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen38Config):
        super().__init__()
        self.cfg = cfg
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = cfg.linear_key_dim, cfg.linear_value_dim
        self.kernel = cfg.linear_conv_kernel_dim
        self.conv_dim = cfg.conv_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.nv))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nv).uniform_(0, 16)))
        self.norm = RMSNormGated(self.dv, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.nv, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.nv, bias=False)

    def causal_conv(self, mixed_qkv, conv_state=None):
        """mixed_qkv ``[B, C, T]`` pre-conv. Returns silu(conv) ``[B, C, T]`` and new conv_state ``[B, C, K-1]``."""
        B, C, T = mixed_qkv.shape
        left = (
            torch.zeros(B, C, self.kernel - 1, dtype=mixed_qkv.dtype)
            if conv_state is None
            else conv_state.to(mixed_qkv.dtype)
        )
        x = torch.cat([left, mixed_qkv], dim=-1)
        new_state = x[:, :, -(self.kernel - 1) :].clone()
        out = F.silu(F.conv1d(x, self.conv1d.weight, None, padding=0, groups=C))
        return out, new_state

    def forward(self, x, conv_state=None, recurrent_state=None, core="chunk"):
        B, T, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(B, T, -1, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        mixed_qkv, new_conv_state = self.causal_conv(mixed_qkv, conv_state)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, -1, self.dk)
        k = k.reshape(B, T, -1, self.dk)
        v = v.reshape(B, T, -1, self.dv)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        rep = self.nv // self.nk
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
        rule = chunk_gated_delta_rule if core == "chunk" else recurrent_gated_delta_rule
        o, new_rec = rule(q, k, v, g, beta, initial_state=recurrent_state)
        o = self.norm(o.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(B, T, -1)
        return self.out_proj(o), new_conv_state, new_rec


# ------------------------------------------------------------------------------------------------
# gated full attention
# ------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: Qwen38Config):
        super().__init__()
        self.cfg = cfg
        self.hd = cfg.head_dim
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        """x ``[B, T, H]``; cos/sin for the T new positions. past_k/v ``[B, Hkv, P, D]`` (post-RoPE)."""
        B, T, _ = x.shape
        q, gate = torch.chunk(self.q_proj(x).view(B, T, -1, self.hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, -1, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, -1, self.hd).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        k_all = k if past_k is None else torch.cat([past_k.to(k.dtype), k], dim=2)
        v_all = v if past_v is None else torch.cat([past_v.to(v.dtype), v], dim=2)
        P = k_all.shape[2] - T
        rep = self.nh // self.nkv
        kk = k_all.repeat_interleave(rep, dim=1)
        vv = v_all.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kk.transpose(2, 3)) * (self.hd**-0.5)
        causal = torch.ones(T, P + T, dtype=torch.bool).tril(diagonal=P)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, vv).transpose(1, 2).reshape(B, T, -1)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out), k, v


# ------------------------------------------------------------------------------------------------
# MLP / layer / model
# ------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: Qwen38Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen38Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = cfg.is_full_attention(layer_idx)
        if self.is_full:
            self.self_attn = Attention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, state=None):
        """state: None (fresh) or the layer's carried state dict. Returns (x, new_state).

        new_state for attention: ``{"k", "v"}`` for the *new* tokens only (the caller appends).
        new_state for GDN: ``{"conv_state", "recurrent_state"}`` after the new tokens.
        """
        state = state or {}
        h = self.input_layernorm(x)
        if self.is_full:
            h, k, v = self.self_attn(h, cos, sin, state.get("k"), state.get("v"))
            new_state = {"k": k, "v": v}
        else:
            h, cs, rs = self.linear_attn(h, state.get("conv_state"), state.get("recurrent_state"))
            new_state = {"conv_state": cs, "recurrent_state": rs}
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_state


class TextModel(nn.Module):
    """embed -> N x DecoderLayer -> final norm (-> lm_head, separately)."""

    def __init__(self, cfg: Qwen38Config, with_lm_head: bool = True):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) if with_lm_head else None

    def forward(self, token_ids, states=None, start_pos: int = 0, return_logits=False):
        """token_ids ``[B, T]``. states: list (one per layer) of carried state or None.

        Returns (final_hidden [B,T,H], states) where attention states hold the *accumulated* K/V.
        """
        T = token_ids.shape[1]
        states = states if states is not None else [None] * len(self.layers)
        x = self.embed_tokens(token_ids)
        cos, sin = rope_cos_sin(self.cfg, torch.arange(start_pos, start_pos + T), dtype=x.dtype)
        new_states = []
        for layer, st in zip(self.layers, states):
            x, ns = layer(x, cos, sin, st)
            if layer.is_full and st is not None:
                ns = {"k": torch.cat([st["k"], ns["k"]], dim=2), "v": torch.cat([st["v"], ns["v"]], dim=2)}
            new_states.append(ns)
        x = self.norm(x)
        if return_logits:
            return self.lm_head(x), new_states
        return x, new_states


# ------------------------------------------------------------------------------------------------
# state_dict helpers (HF checkpoint naming <-> this reference)
# ------------------------------------------------------------------------------------------------
HF_TEXT_PREFIX = "model.language_model."


def hf_key_to_ref(key: str) -> str | None:
    """Map a raw checkpoint key to this reference's naming, or None if not part of the text model."""
    if key == "lm_head.weight":
        return key
    if not key.startswith(HF_TEXT_PREFIX):
        return None  # visual / mtp
    return key[len(HF_TEXT_PREFIX) :]


def init_random_(module: nn.Module, seed: int = 0, std: float = 0.02):
    """Deterministic random init that keeps activations well scaled (for random-weight PCC tests)."""
    gen = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty(p.shape).uniform_(1, 16, generator=gen)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.rand(p.shape, generator=gen) * 2 - 1)
            elif name.endswith("norm.weight") and "linear_attn.norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            elif "norm" in name:  # zero-centred norms: small deviation around 0 => weight ~1
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            elif name.endswith("conv1d.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * (1.0 / math.sqrt(p.shape[-1])))
            elif name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)
    return module
