# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioAttention (HF `Gemma4AudioAttention`, modeling_gemma4.py:257-360): causal-local attention with a
relative-position term, per-dim query scale and a logit softcap.

HF computes it in blocks of 12 queries against a 24-key context; with the HF mask (0 <= q - kv < 12,
key valid) that is exactly a band: query q sees keys q-11 .. q. This module evaluates the same band
on tile-aligned blocks of 32 queries, each against a 64-key context (previous 32 + own 32 keys),
which covers the band for every query of the block:

    q = ClippableLinear_q(x) * (head_dim^-0.5 / ln2 * softplus(per_dim_scale))     fp32, per channel
    k = ClippableLinear_k(x) * ln(1 + e) / ln2;   v = ClippableLinear_v(x)          fp32
    split 8 heads x 128; blocks of 32 rows: Qb [B, 8*nb, 32, 128]
    Kctx, Vctx = concat(previous block, own block)                                  [B, 8*nb, 64, 128]
    ac = Qb Kctx^T                                                                  [B, 8*nb, 32, 64]
    bd = rel_shift(Q E^T): X = Q E^T [.., 32, 96], bd[i, j] = X[i, 32 - i + j]       (Transformer-XL shift)
         E row c = 64 - d holds relative_k_proj(sinusoid(d)) for distance d = q - kv in 0..12, else 0
    logits = tanh((ac + bd) / 50) * 50 + mask          mask: 0 on the band of valid keys, -1e9 elsewhere
    p = softmax(logits) in fp32 (max / exp / sum / reciprocal: the fused ttnn.softmax is less exact)
    out = ClippableLinear_post(concat_heads(p Vctx))

HF replaces masked logits by -1e9 after the softcap; adding -1e9 to a capped logit (|l| <= 50) gives
the same fp32 value, so every row with at least one valid key (every valid query: it sees itself)
matches. Rows of padding queries are not meaningful (HF's are not either); nothing valid reads them.

Contract
    input   x [B, 1, S, 1024] bf16 TILE DRAM (norm_pre_attn output), S a multiple of 32
            mask [B, 1, S, 64] fp32 TILE DRAM (audio_encoder.attention_mask; 0 keep / -1e9 drop)
    output  [B, 1, S, 1024] bf16 TILE DRAM
    Constants (load time): qscale [1,1,1,1024] fp32, E^T [1, 8, 128, 96] fp32 per layer.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.clippable_linear import ClippableLinear
from models.demos.blackhole.embedding_gemma2.tt.audio.config import ATTN_BLOCK, AudioAttentionConfig, ClipBounds
from models.demos.blackhole.embedding_gemma2.tt.audio.linear import audio_linear
from models.demos.blackhole.embedding_gemma2.tt.audio.program_configs import block_bmm_config
from models.demos.blackhole.embedding_gemma2.tt.audio.weights import clip_bounds
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4, compute_kernel_config_norm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight, substate

CTX = 2 * ATTN_BLOCK  # keys per query block: previous block + own block
REL_WIDTH = 3 * ATTN_BLOCK  # columns of Q E^T before the shift


def rel_position_table(cfg: AudioAttentionConfig) -> torch.Tensor:
    """HF `Gemma4AudioRelPositionalEncoding` output [13, hidden] in fp32: rows are distances 12 .. 0."""
    num_timescales = cfg.hidden_size // 2
    inc = math.log(10000.0 / 1.0) / max(num_timescales - 1, 1)
    inv = 1.0 * torch.exp(torch.arange(num_timescales) * -inc)
    pos = torch.arange(cfg.max_past_horizon, -1, -1)[:, None]
    scaled = pos * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).float()


def rel_key_matrix(rel_k_proj: torch.Tensor, cfg: AudioAttentionConfig) -> torch.Tensor:
    """E^T [1, H, head_dim, 96] (fp32): column c = 64 - d holds relative_k_proj(sinusoid(d)), d = 0..12."""
    relk = rel_position_table(cfg) @ rel_k_proj.float().t()  # [13, H*hd]; row p <-> distance 12 - p
    relk = relk.view(-1, cfg.num_heads, cfg.head_dim)
    E = torch.zeros(cfg.num_heads, REL_WIDTH, cfg.head_dim)
    for p in range(relk.shape[0]):
        d = cfg.max_past_horizon - p
        E[:, CTX - d] = relk[p]
    return E.transpose(-1, -2).unsqueeze(0).contiguous()


def attention_mask(num_valid: list[int], seq_len: int, cfg: AudioAttentionConfig) -> torch.Tensor:
    """Host-built additive mask [B, 1, S, 64] (fp32) for the blocked band layout.

    Column j of query q's row is key kv = 32 * (q // 32) - 32 + j. Kept (0.0) when
    0 <= q - kv < max_past_horizon (HF sliding_window_mask_function, left window 12) and
    0 <= kv < num_valid[b]; dropped (invalid_logit) otherwise.
    """
    q = torch.arange(seq_len)[:, None]
    kv = (q // ATTN_BLOCK) * ATTN_BLOCK - ATTN_BLOCK + torch.arange(CTX)[None, :]
    dist = q - kv
    band = (dist >= 0) & (dist < cfg.max_past_horizon) & (kv >= 0)
    out = torch.full((len(num_valid), 1, seq_len, CTX), cfg.invalid_logit)
    for b, n in enumerate(num_valid):
        out[b, 0][band & (kv < n)] = 0.0
    return out


def block_context(x: ttnn.Tensor) -> ttnn.Tensor:
    """[B, H, S, D] -> [B, H*nb, 64, D]: per 32-row block, the previous block's rows (zeros for the
    first block) followed by its own rows. Device ops only; S must be a multiple of 32."""
    B, H, S, D = x.shape
    nb = S // ATTN_BLOCK
    dram = ttnn.DRAM_MEMORY_CONFIG
    # A device-side zero block (x is finite). Not deallocating `first`: for S == 32 the slice is x itself.
    first = ttnn.slice(x, (0, 0, 0, 0), (B, H, ATTN_BLOCK, D))
    zeros = ttnn.multiply(first, 0.0, memory_config=dram)
    if nb > 1:
        head = ttnn.slice(x, (0, 0, 0, 0), (B, H, S - ATTN_BLOCK, D))
        prev = ttnn.concat([zeros, head], dim=2, memory_config=dram)
        ttnn.deallocate(head)
        ttnn.deallocate(zeros)
    else:
        prev = zeros
    prev_b = ttnn.reshape(prev, (B, H * nb, ATTN_BLOCK, D))
    own_b = ttnn.reshape(x, (B, H * nb, ATTN_BLOCK, D))
    ctx = ttnn.concat([prev_b, own_b], dim=2, memory_config=dram)
    ttnn.deallocate(prev)
    return ctx


class AudioAttention(LightweightModule):
    def __init__(
        self,
        config: AudioAttentionConfig,
        wqkv: ttnn.Tensor,
        qkv_bounds: tuple[ClipBounds, ClipBounds, ClipBounds],
        post: ClippableLinear,
        qscale: ttnn.Tensor,
        rel_keys_t: ttnn.Tensor,
    ):
        super().__init__()
        self.config = config
        self.wqkv = wqkv
        self.qkv_bounds = qkv_bounds
        self.post = post
        self.qscale = qscale
        self.rel_keys_t = rel_keys_t
        self.compute_kernel_config = compute_kernel_config_norm()
        self.proj_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioAttentionConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "audio.self_attn",
        act_dtype=ttnn.bfloat16,
    ) -> "AudioAttention":
        """`state_dict` holds the HF `layers.<i>.self_attn.` sub-state (`.linear.` folded). q/k/v are
        always fp32; `act_dtype` is the output dtype of `post`, see AudioModel.

        q / k / v run as one fused projection (weights concatenated to [1024, 3072]) after one input
        clamp. That needs identical q / k / v input bounds, which holds in all 12 layers of
        google/embeddinggemma-2 (E5c `bounds.py`); other bounds raise."""
        H = config.hidden_size
        kw = dict(cache_dir=cache_dir)
        f32 = ttnn.float32
        subs = {p: substate(state_dict, f"{p}_proj") for p in "qkv"}
        for p, sd in subs.items():
            if tuple(sd["weight"].shape) != (H, H):
                raise ValueError(f"{name}.{p}_proj: expected weight {(H, H)}, got {tuple(sd['weight'].shape)}")
        bounds = tuple(clip_bounds(subs[p], f"{name}.{p}_proj") for p in "qkv")
        if len({(b.input_min, b.input_max) for b in bounds}) != 1:
            raise ValueError(f"{name}: q / k / v input clip bounds differ {bounds}; the fused QKV needs them equal")
        wqkv = torch.cat([subs[p]["weight"] for p in "qkv"], dim=0)  # [3H, H]
        wqkv = as_weight(linear_weight(wqkv), device, cache_dir=cache_dir, name=f"{name}.wqkv")
        post = ClippableLinear.from_state_dict(
            substate(state_dict, "post"), H, H, device, name=f"{name}.post", out_dtype=act_dtype, **kw
        )
        pds = state_dict["per_dim_scale"]
        if tuple(pds.shape) != (config.head_dim,):
            raise ValueError(f"{name}.per_dim_scale: expected ({config.head_dim},), got {tuple(pds.shape)}")
        # per head-dim scale, broadcast over the heads of the [B, H, S, hd] layout
        qscale = config.q_scale * torch.nn.functional.softplus(pds.float())
        qscale = as_weight(
            qscale.reshape(1, 1, 1, config.head_dim), device, cache_dir=cache_dir, name=f"{name}.qscale_hd", dtype=f32
        )
        relw = state_dict["relative_k_proj.weight"]
        if tuple(relw.shape) != (H, H):
            raise ValueError(f"{name}.relative_k_proj: expected {(H, H)}, got {tuple(relw.shape)}")
        rel_t = as_weight(
            rel_key_matrix(relw, config), device, cache_dir=cache_dir, name=f"{name}.rel_keys_t", dtype=f32
        )
        return cls(config, wqkv, bounds, post, qscale, rel_t)

    # ------------------------------------------------------------------ helpers
    def _qkv_heads(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """x [B, 1, S, 1024] -> scaled q, scaled k, v, each [B, H, S, hd] fp32.

        q = clamp_q(x' Wq) * qscale, k = clamp_k(x' Wk) * k_scale, v = clamp_v(x' Wv) with
        x' = clamp_in(x): one fused matmul, one head split, and each output clamp applied as the
        SFPU pre-activation of its scale multiply (same ops, same order as separate clamp + mul)."""
        cfg = self.config
        dram = ttnn.DRAM_MEMORY_CONFIG
        bq, bk, bv = self.qkv_bounds
        xc = ttnn.clamp(x, min=bq.input_min, max=bq.input_max, memory_config=dram)
        qkv = audio_linear(
            xc, self.wqkv, dtype=ttnn.float32, compute_kernel_config=self.proj_kernel_config, memory_config=dram
        )
        ttnn.deallocate(xc)
        q_r, k_r, v_r = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=cfg.num_heads, num_kv_heads=cfg.num_heads, transpose_k_heads=False, memory_config=dram
        )
        ttnn.deallocate(qkv)

        def clamp_op(b):
            return ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, b.output_min, b.output_max)

        q = ttnn.multiply(q_r, self.qscale, input_tensor_a_activations=[clamp_op(bq)], memory_config=dram)
        ttnn.deallocate(q_r)
        k = ttnn.multiply(k_r, cfg.k_scale, input_tensor_a_activations=[clamp_op(bk)], memory_config=dram)
        ttnn.deallocate(k_r)
        v = ttnn.clamp(v_r, min=bv.output_min, max=bv.output_max, memory_config=dram)
        ttnn.deallocate(v_r)
        return q, k, v

    def _rel_shift(self, X: ttnn.Tensor) -> ttnn.Tensor:
        """X [B, M, 32, 96] -> [B, M, 32, 64] with out[i, j] = X[i, 32 - i + j] (row-major view trick)."""
        B, M, R, W = X.shape
        x = ttnn.to_layout(X, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, M, W, R))
        x = ttnn.slice(x, (0, 0, 1, 0), (B, M, W, R))
        x = ttnn.reshape(x, (B, M, R, W - 1))
        x = ttnn.slice(x, (0, 0, 0, 0), (B, M, R, CTX))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _softmax(self, s: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        m = ttnn.max(s, dim=-1, keepdim=True)
        z = ttnn.subtract(s, m, memory_config=dram)
        ttnn.deallocate(m)
        e = ttnn.exp(z, fast_and_approximate_mode=False, memory_config=dram)
        ttnn.deallocate(z)
        denom = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.compute_kernel_config)
        r = ttnn.reciprocal(denom)
        ttnn.deallocate(denom)
        p = ttnn.multiply(e, r, memory_config=dram)
        ttnn.deallocate(e)
        ttnn.deallocate(r)
        return p

    # ------------------------------------------------------------------ forward
    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        dram = ttnn.DRAM_MEMORY_CONFIG
        ckc = self.compute_kernel_config
        B, _, S, _ = x.shape
        if S % ATTN_BLOCK != 0:
            raise ValueError(f"audio attention needs S % {ATTN_BLOCK} == 0, got {S}")
        nb, H, hd = S // ATTN_BLOCK, cfg.num_heads, cfg.head_dim

        q, k, v = self._qkv_heads(x)

        # relative-position term on the [S, 96] layout, then shifted into the 64-key context
        rel_t = self.rel_keys_t if B == 1 else ttnn.repeat(self.rel_keys_t, ttnn.Shape([B, 1, 1, 1]))
        X = ttnn.matmul(q, rel_t, compute_kernel_config=ckc, memory_config=dram)
        if B != 1:
            ttnn.deallocate(rel_t)
        Xb = ttnn.reshape(X, (B, H * nb, ATTN_BLOCK, REL_WIDTH))
        bd = self._rel_shift(Xb)
        ttnn.deallocate(X)

        qb = ttnn.reshape(q, (B, H * nb, ATTN_BLOCK, hd))
        k_ctx = block_context(k)
        v_ctx = block_context(v)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        dev = x.device()
        ac = ttnn.matmul(
            qb,
            k_ctx,
            transpose_b=True,
            program_config=block_bmm_config(dev, ATTN_BLOCK, hd, CTX),
            compute_kernel_config=ckc,
            memory_config=dram,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_ctx)

        # tanh((ac + bd) / cap) * cap: the softcap runs as post-activations of the add (fp32 SFPU,
        # same ops in the same order as separate multiply / tanh / multiply ops)
        cap = cfg.logit_cap
        softcap = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / cap),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH, 0.0),  # 0: accurate (fast_and_approximate_mode=False)
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, cap),
        ]
        capped = ttnn.add(ac, bd, activations=softcap, memory_config=dram)
        ttnn.deallocate(ac)
        ttnn.deallocate(bd)
        capped = ttnn.reshape(capped, (B, H, S, CTX))
        masked = ttnn.add(capped, mask, memory_config=dram)
        ttnn.deallocate(capped)
        probs = self._softmax(masked)
        ttnn.deallocate(masked)
        probs = ttnn.reshape(probs, (B, H * nb, ATTN_BLOCK, CTX))

        o = ttnn.matmul(
            probs,
            v_ctx,
            program_config=block_bmm_config(dev, ATTN_BLOCK, CTX, hd),
            compute_kernel_config=ckc,
            memory_config=dram,
        )  # [B, H*nb, 32, hd]
        ttnn.deallocate(probs)
        ttnn.deallocate(v_ctx)
        o = ttnn.reshape(o, (B, H, S, hd))
        o_cat = ttnn.experimental.nlp_concat_heads(o, memory_config=dram)  # [B, 1, S, H*hd]
        ttnn.deallocate(o)
        out = self.post(o_cat)
        ttnn.deallocate(o_cat)
        return out
