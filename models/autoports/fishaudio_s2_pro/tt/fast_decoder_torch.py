"""Torch (CPU) implementation of the S2 Pro fast codebook decoder, faithful to fish-speech's DualAR
fast path: 4 pre-norm blocks (no q/k norm), interleaved RoPE (base 1e6) over positions 0..9, a per-layer
KV cache of num_codebooks positions, explicit-matmul attention, SwiGLU, final RMSNorm, 4096-way head.

Used by Phase A end-to-end generation and as the reference for the TTNN fast decoder (Phase B).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from models.autoports.fishaudio_s2_pro.config import S2Config, TowerConfig


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float, dtype=torch.bfloat16) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1).to(dtype)  # (seq, n_elem/2, 2)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, D); freqs_cis: (S, D/2, 2). Interleaved-pair (Meta) convention, computed in fp32."""
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    fc = freqs_cis.float().view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    out = torch.stack(
        [
            xshaped[..., 0] * fc[..., 0] - xshaped[..., 1] * fc[..., 1],
            xshaped[..., 1] * fc[..., 0] + xshaped[..., 0] * fc[..., 1],
        ],
        -1,
    )
    return out.flatten(3).type_as(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


class FastBlock(torch.nn.Module):
    def __init__(self, cfg: TowerConfig, w: Dict[str, torch.Tensor], prefix: str):
        super().__init__()
        self.cfg = cfg
        g = lambda n: torch.nn.Parameter(w[prefix + n], requires_grad=False)
        if prefix + "attention.wqkv.weight" in w:
            self.wqkv = g("attention.wqkv.weight")
        else:
            self.wqkv = torch.nn.Parameter(
                torch.cat(
                    [
                        w[prefix + "attention.wq.weight"],
                        w[prefix + "attention.wk.weight"],
                        w[prefix + "attention.wv.weight"],
                    ],
                    0,
                ),
                requires_grad=False,
            )
        self.wo = g("attention.wo.weight")
        self.w1, self.w2, self.w3 = (
            g("feed_forward.w1.weight"),
            g("feed_forward.w2.weight"),
            g("feed_forward.w3.weight"),
        )
        self.attention_norm = RMSNorm(w[prefix + "attention_norm.weight"], cfg.norm_eps)
        self.ffn_norm = RMSNorm(w[prefix + "ffn_norm.weight"], cfg.norm_eps)
        self.k_cache = torch.zeros(1, cfg.n_local_heads, 32, cfg.head_dim, dtype=self.wqkv.dtype)
        self.v_cache = torch.zeros_like(self.k_cache)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, pos: int) -> torch.Tensor:
        cfg = self.cfg
        B, S, _ = x.shape  # S == 1 in generation
        h = self.attention_norm(x)
        qkv = F.linear(h, self.wqkv)
        q, k, v = qkv.split([cfg.q_dim, cfg.kv_dim, cfg.kv_dim], dim=-1)
        q = q.view(B, S, cfg.n_head, cfg.head_dim)
        k = k.view(B, S, cfg.n_local_heads, cfg.head_dim)
        v = v.view(B, S, cfg.n_local_heads, cfg.head_dim)
        q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, S, D)
        self.k_cache[:, :, pos : pos + S] = k
        self.v_cache[:, :, pos : pos + S] = v
        k_all, v_all = self.k_cache[:, :, : pos + S], self.v_cache[:, :, : pos + S]
        rep = cfg.n_head // cfg.n_local_heads
        k_all, v_all = k_all.repeat_interleave(rep, dim=1), v_all.repeat_interleave(rep, dim=1)
        att = (q.float() @ k_all.float().transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        if S > 1:
            causal = torch.tril(torch.ones(S, pos + S, dtype=torch.bool), diagonal=pos)
            att = att.masked_fill(~causal, float("-inf"))
        att = torch.softmax(att, dim=-1).type_as(q)
        y = (att @ v_all).transpose(1, 2).reshape(B, S, cfg.q_dim)
        x = x + F.linear(y, self.wo)
        h = self.ffn_norm(x)
        return x + F.linear(F.silu(F.linear(h, self.w1)) * F.linear(h, self.w3), self.w2)


class TorchFastDecoder(torch.nn.Module):
    """codes for one frame from the slow tower's (post-norm) hidden state."""

    def __init__(self, fast_sd: Dict[str, torch.Tensor], cfg: S2Config, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        fc = cfg.fast
        w = {k: v.to(dtype) for k, v in fast_sd.items()}
        self.embeddings = torch.nn.Parameter(w["embeddings.weight"], requires_grad=False)
        self.layers = torch.nn.ModuleList([FastBlock(fc, w, f"layers.{i}.") for i in range(fc.n_layer)])
        self.norm = RMSNorm(w["norm.weight"], fc.norm_eps)
        self.output = torch.nn.Parameter(w["output.weight"], requires_grad=False)
        self.freqs_cis = precompute_freqs_cis(cfg.num_codebooks, fc.head_dim, fc.rope_base, dtype=torch.bfloat16)
        self.dtype = dtype

    @torch.inference_mode()
    def step(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """x: (D,) or (1,1,D). Returns logits (codebook_size,) float32."""
        x = x.to(self.dtype).view(1, 1, -1)
        fc = self.freqs_cis[pos : pos + 1]
        for layer in self.layers:
            x = layer(x, fc, pos)
        return F.linear(self.norm(x), self.output)[0, -1].float()

    @torch.inference_mode()
    def frame(self, hidden: torch.Tensor, code0: int, choose, record: Optional[List] = None) -> List[int]:
        """Full frame: pos-0 pass on the hidden state (output discarded), then codebooks 1..9.
        `choose(logits, codebook_idx) -> int` decides each code (argmax or sampler)."""
        self.step(hidden, 0)
        h = self.embeddings[code0]
        out = []
        for i in range(1, self.cfg.num_codebooks):
            logits = self.step(h, i)
            if record is not None:
                record.append(logits)
            c = int(choose(logits, i))
            out.append(c)
            h = self.embeddings[c]
        return out
