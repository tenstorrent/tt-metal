# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for Hy4-preview's attention sublayer (gated MLA with a sink).

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py`` HYV4Attention
with the eager attention core (eager_attention_forward), minus the cache and the indexer: which
positions a query may attend to arrives as an additive mask, so the DSA top-k selection is the
caller's (tenstorrent/tt-metal#65887, #65888).

    q_resid = RMSNorm(x W_qa)                      [S, 2048]
    q       = q_resid W_qb  -> per head (q_nope 192 | q_rope 64), RoPE on q_rope
    kv      = x W_kva       -> (kv_pass 512 | k_rope 64); k_pass = RMSNorm(kv_pass), RoPE on k_rope
    k, v    = k_pass W_kvb  -> per head (k_nope 192 | v 256); key = (k_nope | k_rope), k_rope shared by all heads
    probs   = softmax over the allowed keys AND one extra per-head sink logit; the sink's share is dropped
    out     = ((probs @ v) * sigmoid(x W_gate)) W_o

What the cache stores per token is the latent (k_pass | k_rope), 576 values shared by all heads.

Differences from GLM-5.x / DeepSeek MLA: the elementwise output gate (``gated_mla``) and the
learnable per-head sink (``learnable_sink``). Both norms use eps 1e-6 (HYV4RMSNorm's default), not
the model's ``rms_norm_eps``. RoPE is the plain half-split form, theta 1e7, no YaRN.

Checkpoint names under ``model.layers.{L}.self_attn.`` (the HF module calls the last two
``gate_proj`` and ``sinks``):
    q_a_proj.weight  q_a_layernorm.weight  q_b_proj.weight  kv_a_proj_with_mqa.weight
    kv_a_layernorm.weight  kv_b_proj.weight  o_proj.weight  linear_gate.weight  learnable_sink_param
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    """Attention hyperparameters. Defaults are the tencent/Hy4-preview config.json values."""

    hidden_size: int = 6144
    num_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    norm_eps: float = 1e-6  # q_a_layernorm / kv_a_layernorm (HYV4RMSNorm default)
    rope_theta: float = 1e7

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_b_head_dim(self) -> int:
        # kv_b_proj's output per head: (k_nope | v)
        return self.qk_nope_head_dim + self.v_head_dim

    @property
    def latent_dim(self) -> int:
        # what the cache stores per token: (k_pass | k_rope)
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def scaling(self) -> float:
        return self.qk_head_dim**-0.5


# Which dims of the roped part form a rotated pair. ONE switch for the reference, the frames (tt/frames.py) and so
# for everything built on them:
#   "half_split"   pairs (i, i + d / 2): HF's modeling_hy_v4.py as it runs here; every golden recorded so far
#   "interleaved"  pairs (2 i, 2 i + 1): vLLM's hy_v4 and what the checkpoint's rows say (the reference-health probes,
#                  fused/README.md section 49)
# The default stays "half_split" until the goldens are recorded again: the two must change together.
ROPE_PAIRING_ENV = "HY4_ROPE_PAIRING"
ROPE_PAIRINGS = ("half_split", "interleaved")


def rope_pairing(pairing: str | None = None) -> str:
    """``pairing`` if given, else the process's (HY4_ROPE_PAIRING, read at every call)."""
    import os

    value = pairing or os.environ.get(ROPE_PAIRING_ENV) or "half_split"
    assert value in ROPE_PAIRINGS, f"{ROPE_PAIRING_ENV}={value!r}: one of {ROPE_PAIRINGS}"
    return value


def rope_cos_sin(positions: torch.Tensor, cfg: AttentionConfig, pairing: str | None = None):
    """HYV4RotaryEmbedding: positions [S] -> (cos, sin), each [S, qk_rope_head_dim], fp32: every frequency on both
    dims of its pair (half_split: cat(freqs, freqs); interleaved: each frequency twice in a row)."""
    dim = cfg.qk_rope_head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = positions[..., None].float() * inv_freq
    if rope_pairing(pairing) == "half_split":
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        emb = freqs.repeat_interleave(2, dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """(-x1, x0, -x3, x2, ...): the quarter turn of every pair (2 i, 2 i + 1)."""
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pairing: str | None = None) -> torch.Tensor:
    """RoPE in the process's pairing (rope_pairing; cos / sin from rope_cos_sin in the same). x: [..., S, dim];
    cos / sin: [S, dim]."""
    rotated = rotate_half(x) if rope_pairing(pairing) == "half_split" else rotate_pairs(x)
    return x * cos + rotated * sin


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HYV4RMSNorm: normalised in fp32, cast back, then scaled by the weight."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def causal_mask(seq_len: int, dtype=torch.float32) -> torch.Tensor:
    """Additive [S, S] mask: 0 where key <= query, the dtype's most negative value elsewhere."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)


class Hy4Attention(nn.Module):
    """One attention sublayer on a whole prompt. x: [S, hidden] -> [S, hidden].

    The caller applies the layer's pre-norm (``input_layernorm``) before and iHC's hc_post after.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.q_a_proj = nn.Parameter(torch.empty(c.q_lora_rank, c.hidden_size))
        self.q_a_layernorm = nn.Parameter(torch.ones(c.q_lora_rank))
        self.q_b_proj = nn.Parameter(torch.empty(c.num_heads * c.qk_head_dim, c.q_lora_rank))
        self.kv_a_proj_with_mqa = nn.Parameter(torch.empty(c.latent_dim, c.hidden_size))
        self.kv_a_layernorm = nn.Parameter(torch.ones(c.kv_lora_rank))
        self.kv_b_proj = nn.Parameter(torch.empty(c.num_heads * c.kv_b_head_dim, c.kv_lora_rank))
        self.o_proj = nn.Parameter(torch.empty(c.hidden_size, c.num_heads * c.v_head_dim))
        self.linear_gate = nn.Parameter(torch.empty(c.num_heads * c.v_head_dim, c.hidden_size))
        self.learnable_sink_param = nn.Parameter(torch.zeros(c.num_heads))
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "linear_gate"):
            nn.init.normal_(getattr(self, name), std=0.006)

    PARAMETER_NAMES = (
        "q_a_proj",
        "q_a_layernorm",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_a_layernorm",
        "kv_b_proj",
        "o_proj",
        "linear_gate",
        "learnable_sink_param",
    )

    def weights(self) -> dict:
        """The nine parameters by checkpoint name, as the device module takes them."""
        return {name: getattr(self, name).detach() for name in self.PARAMETER_NAMES}

    def latent(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x: [S, hidden] -> (q_resid [S, q_lora], k_pass [S, kv_lora], k_rope [S, rope]).

        (k_pass | k_rope) is the per-token cache entry; q_resid also feeds the indexer.
        """
        c = self.cfg
        q_resid = rms_norm(F.linear(x, self.q_a_proj), self.q_a_layernorm, c.norm_eps)
        kv_pass, k_rope = F.linear(x, self.kv_a_proj_with_mqa).split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        k_pass = rms_norm(kv_pass, self.kv_a_layernorm, c.norm_eps)
        return q_resid, k_pass, apply_rope(k_rope, cos, sin)

    def queries(self, q_resid: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """q_resid [S, q_lora] -> [H, S, qk_head_dim], RoPE applied to the trailing rope dims."""
        c = self.cfg
        q = F.linear(q_resid, self.q_b_proj).view(-1, c.num_heads, c.qk_head_dim).transpose(0, 1)
        q_pass, q_rope = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        return torch.cat((q_pass, apply_rope(q_rope, cos, sin)), dim=-1)

    def keys_values(self, k_pass: torch.Tensor, k_rope: torch.Tensor):
        """Latent -> (keys [H, T, qk_head_dim], values [H, T, v_head_dim])."""
        c = self.cfg
        kv = F.linear(k_pass, self.kv_b_proj).view(-1, c.num_heads, c.kv_b_head_dim).transpose(0, 1)
        k_nope, v = kv.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)
        keys = torch.cat((k_nope, k_rope.unsqueeze(0).expand(c.num_heads, -1, -1)), dim=-1)
        return keys, v

    def core(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """eager_attention_forward: [H, S, .] x [H, T, .] -> [H, S, v_head_dim].

        ``mask`` is additive, broadcastable to [H, S, T]. The sink is one more logit per head that
        takes probability mass and contributes no value.
        """
        logits = torch.matmul(q, keys.transpose(-1, -2)) * self.cfg.scaling + mask
        sinks = self.learnable_sink_param.to(logits.dtype).reshape(-1, 1, 1).expand(-1, q.shape[-2], -1)
        combined = torch.cat([logits, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
        return torch.matmul(probs.to(values.dtype), values)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """x: [S, hidden] -> sigmoid(x W_gate), [S, H * v_head_dim], head-major."""
        return torch.sigmoid(F.linear(x, self.linear_gate))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [S, hidden]; cos / sin: [S, rope]; mask: additive [S, S]. Returns [S, hidden]."""
        q_resid, k_pass, k_rope = self.latent(x, cos, sin)
        q = self.queries(q_resid, cos, sin)
        keys, values = self.keys_values(k_pass, k_rope)
        attn = self.core(q, keys, values, mask)  # [H, S, v]
        attn = attn.transpose(0, 1).reshape(x.shape[0], -1)  # [S, H*v], head-major
        return F.linear(attn * self.gate(x), self.o_proj)
