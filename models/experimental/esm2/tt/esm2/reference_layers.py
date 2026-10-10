# SPDX-License-Identifier: MIT
"""CPU (PyTorch) reference implementation of ESM-2 for esm2_t33_650M_UR50D.

Mirrors the TTNN op map in RESEARCH.md one-to-one so the same module graph is
validated offline before it lands on device. Inference-only: no dropout RNG.
Semantics verified against transformers 5.12.1 models/esm/modeling_esm.py;
position semantics corrected from the installed oracle source after the
bringup batch-pad failure (see position_ids_from_input_ids).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import Esm2TTConfig

# ---------------------------------------------------------------- helpers


def position_ids_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Plain 0-based positions (arange), matching the oracle exactly.

    Measured correction: the
    installed transformers EsmModel computes ``position_ids = arange(seq_len)``
    for the rotary path; the pad-aware fairseq-style cumsum (pads ->
    padding_idx) belongs to its unused absolute-position path. The two schemes
    give identical attention for real tokens (RoPE scores are invariant to a
    constant global position offset) but differ at pad rows: on the padded
    bringup batch row the cumsum scheme misses the oracle by 0.371 hidden
    NRMSE at pad positions (real positions 4.2e-4) vs 0.013 for arange.
    Signature kept for drop-in compatibility with existing callers.
    """
    return torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).expand_as(input_ids).contiguous()


def additive_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """[B,L] {0,1} -> [B,1,1,L] additive fp32 (0 keep / finfo.min drop)."""
    am = attention_mask[:, None, None, :].to(torch.float32)
    return (1.0 - am) * torch.finfo(torch.float32).min


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryTables:
    """cos/sin [B,L,head_dim] fp32, computed once per (position_ids, L)."""

    def __init__(self, config: Esm2TTConfig, device=None):
        self.head_dim = config.head_dim
        self.base = config.rotary_base
        self.device = device
        self._cache: dict = {}

    def cos_sin(self, position_ids: torch.Tensor):
        key = (tuple(position_ids.shape), int(position_ids.min()), int(position_ids.max()), position_ids.sum().item())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = torch.einsum("bl,f->blf", position_ids.to(torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,L,head_dim]
        out = (emb.cos(), emb.sin())
        self._cache[key] = out
        return out


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k [B,H,L,D]; cos/sin [B,L,D] fp32 -> broadcast over heads."""
    cos = cos.to(q.dtype).unsqueeze(1)
    sin = sin.to(q.dtype).unsqueeze(1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------- modules


class Esm2Embeddings(torch.nn.Module):
    """Word embedding + ESM token-dropout semantics (no position table)."""

    def __init__(self, config: Esm2TTConfig, weight: torch.Tensor):
        super().__init__()
        self.weight = weight  # [vocab, hidden] fp32, tied with LM head decoder
        self.config = config

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        h = F.embedding(input_ids, self.weight)
        if cfg.token_dropout:
            h = h.masked_fill((input_ids == cfg.mask_token_id).unsqueeze(-1), 0.0)
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == cfg.mask_token_id).sum(-1).float() / src_lengths
            h = h * (1.0 - cfg.mask_ratio_train) / (1.0 - mask_ratio_observed[:, None, None])
        return h  # [B,L,hidden] fp32


class Esm2SelfAttention(torch.nn.Module):
    """MHA with ESM quirks: q scaled by d^-0.5 BEFORE rotary; additive mask."""

    def __init__(self, config: Esm2TTConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.q = torch.nn.Linear(self.hidden, self.hidden)
        self.k = torch.nn.Linear(self.hidden, self.hidden)
        self.v = torch.nn.Linear(self.hidden, self.hidden)

    def forward(self, x_ln, attn_bias, cos, sin):
        B, L, _ = x_ln.shape
        q = self.q(x_ln).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
        k = self.k(x_ln).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_ln).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = q * self.head_dim**-0.5
        q, k = apply_rotary(q, k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) + attn_bias  # fp32
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # [B,H,L,D]
        out = out.transpose(1, 2).reshape(B, L, self.hidden)
        return out


class Esm2Layer(torch.nn.Module):
    """Pre-LN residual block: x + FFN(LN2(x + OUT(Attn(LN1(x))))) exact."""

    def __init__(self, config: Esm2TTConfig):
        super().__init__()
        self.attn = Esm2SelfAttention(config)
        self.attn_out = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.ln_attn = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_ffn = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x, attn_bias, cos, sin):
        x_ln = self.ln_attn(x)
        a = self.attn(x_ln, attn_bias, cos, sin)
        x = x + self.attn_out(a)
        z = self.ln_ffn(x)
        return x + self.ffn2(F.gelu(self.ffn1(z)))


class Esm2Model(torch.nn.Module):
    """Full masked-LM model: embeddings -> 33 pre-LN layers -> final LN -> head."""

    def __init__(self, config: Esm2TTConfig, weights: dict | None = None):
        super().__init__()
        self.config = config
        emb_w = torch.zeros(config.vocab_size, config.hidden_size)
        self.embeddings = Esm2Embeddings(config, emb_w)
        self.layers = torch.nn.ModuleList(Esm2Layer(config) for _ in range(config.num_hidden_layers))
        self.final_ln = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_ln = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_bias = torch.zeros(config.vocab_size)
        self.rotary = RotaryTables(config)
        if weights is not None:
            self.load_canonical(weights)

    # ---- weight loading (canonical names; see loader.py) ----
    def load_canonical(self, w: dict):
        self.embeddings.weight = torch.nn.Parameter(w["embeddings.word_embeddings.weight"].float())
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}."
            layer.attn.q.weight.data = w[p + "attn.q.weight"].float()
            layer.attn.q.bias.data = w[p + "attn.q.bias"].float()
            layer.attn.k.weight.data = w[p + "attn.k.weight"].float()
            layer.attn.k.bias.data = w[p + "attn.k.bias"].float()
            layer.attn.v.weight.data = w[p + "attn.v.weight"].float()
            layer.attn.v.bias.data = w[p + "attn.v.bias"].float()
            layer.attn_out.weight.data = w[p + "attn_out.weight"].float()
            layer.attn_out.bias.data = w[p + "attn_out.bias"].float()
            layer.ln_attn.weight.data = w[p + "ln_attn.weight"].float()
            layer.ln_attn.bias.data = w[p + "ln_attn.bias"].float()
            layer.ln_ffn.weight.data = w[p + "ln_ffn.weight"].float()
            layer.ln_ffn.bias.data = w[p + "ln_ffn.bias"].float()
            layer.ffn1.weight.data = w[p + "ffn1.weight"].float()
            layer.ffn1.bias.data = w[p + "ffn1.bias"].float()
            layer.ffn2.weight.data = w[p + "ffn2.weight"].float()
            layer.ffn2.bias.data = w[p + "ffn2.bias"].float()
        self.final_ln.weight.data = w["final_ln.weight"].float()
        self.final_ln.bias.data = w["final_ln.bias"].float()
        self.lm_dense.weight.data = w["lm.dense.weight"].float()
        self.lm_dense.bias.data = w["lm.dense.bias"].float()
        self.lm_ln.weight.data = w["lm.ln.weight"].float()
        self.lm_ln.bias.data = w["lm.ln.bias"].float()
        self.lm_bias = w["lm.bias"].float()

    # ---- forward ----
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        cfg = self.config
        h = self.embeddings(input_ids, attention_mask)
        pos = position_ids_from_input_ids(input_ids, cfg.pad_token_id)
        cos, sin = self.rotary.cos_sin(pos)
        attn_bias = additive_attention_mask(attention_mask)
        for layer in self.layers:
            h = layer(h, attn_bias, cos, sin)
        hidden = self.final_ln(h)
        x = F.gelu(self.lm_dense(hidden))
        x = self.lm_ln(x)
        logits = torch.nn.functional.linear(x, self.embeddings.weight) + self.lm_bias
        return logits, hidden  # [B,L,33], [B,L,1280] fp32
