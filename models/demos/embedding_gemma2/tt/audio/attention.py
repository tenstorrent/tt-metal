# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioAttention (HF `Gemma4AudioAttention`, modeling_gemma4.py:257-360): causal-local attention with a
relative-position term, per-dim query scale and a logit softcap.

HF computes it in blocks of 12 queries against a 24-key context; with the HF mask (0 <= q - kv < 12,
key valid) that is exactly a band: query q sees keys q-11 .. q. This module evaluates the same band
on tile-aligned blocks of 32 queries, each against a 64-key context (previous 32 + own 32 keys),
which covers the band for every query of the block:

    x' = clamp_in(x)  (q / k / v share the input bounds); qkv = x' [Wq | Wk | Wv]      one matmul, fp32
    split 8 heads x 128 (nlp_create_qkv_heads), then
    q = clamp_q(.) * (head_dim^-0.5 / ln2 * softplus(per_dim_scale))                 fp32, per channel
    k = clamp_k(.) * ln(1 + e) / ln2;   v = clamp_v(.)                                fp32
    blocks of 32 rows: Qb [B, 8*nb, 32, 128]
    Kctx, Vctx = concat(previous block, own block)                                  [B, 8*nb, 64, 128]
    ac = Qb Kctx^T                                                                  [B, 8*nb, 32, 64]
    bd = rel_shift(Q E^T): X = Q E^T [.., 32, 32], bd[i, j] = X[i, j - i - 20]       (Transformer-XL shift,
         one gather); E row c = 12 - d holds relative_k_proj(sinusoid(d)) for distance d = q - kv in 0..12
    logits = tanh((ac + bd) / 50) * 50 + mask          mask: 0 on the band of valid keys, -1e9 elsewhere
    p = softmax(logits) in fp32 (max / exp / sum / reciprocal: the fused ttnn.softmax is less exact)
    out = ClippableLinear_post(concat_heads(p Vctx))
E5c fuses unary steps into neighbouring ops (clamps into matmuls / binary ops, softcap and exp /
reciprocal as SFPU activations) without changing the op order; outputs are bit-identical to E3
except where noted in doc/optimized_audio/README.md.

HF replaces masked logits by -1e9 after the softcap; adding -1e9 to a capped logit (|l| <= 50) gives
the same fp32 value, so every row with at least one valid key (every valid query: it sees itself)
matches. Rows of padding queries are not meaningful (HF's are not either); nothing valid reads them.

Contract
    input   x [B, 1, S, 1024] bf16 TILE, DRAM or L1 (norm_pre_attn output), S a multiple of 32
            mask [B, 1, S, 64] fp32 TILE DRAM (audio_encoder.attention_mask; 0 keep / -1e9 drop)
    output  [B, 1, S, 1024] bf16 TILE, L1 when it fits (config.audio_act_memory_config) else DRAM;
            the intermediates follow the same rule (the fused QKV output with a 96 KB-per-bank cap)
    Constants (load time): Wqkv [1,1,1024,3072] bf16, qscale [1,1,1,128] fp32, E^T [1, 8, 128, 32] fp32
    per layer; the shift index [1,1,32,64] uint16 (shared).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.audio.clippable_linear import ClippableLinear
from models.demos.embedding_gemma2.tt.audio.config import (
    ATTN_BLOCK,
    WIDE_L1_BYTES_PER_BANK,
    AudioAttentionConfig,
    ClipBounds,
    audio_act_memory_config,
)
from models.demos.embedding_gemma2.tt.audio.linear import audio_linear
from models.demos.embedding_gemma2.tt.audio.program_configs import block_bmm_config
from models.demos.embedding_gemma2.tt.audio.weights import clip_bounds
from models.demos.embedding_gemma2.tt.precision import AudioPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight, substate

CTX = 2 * ATTN_BLOCK  # keys per query block: previous block + own block
REL_WIDTH = ATTN_BLOCK  # columns of Q E^T before the shift (column c = max_past_horizon - d)

# per_core_M (32-row blocks per work unit) of the rel-pos Q E^T block matmul: E5c rule = the largest of
# 1 / 2 / 4 dividing the block count. Per-grid tuned entries REL_PM_TUNED[grid][S] (batch 1, E10.1,
# gated with the audio matmul table: EG2_AUDIO_MM_TUNED); REL_PM_OVERRIDES[S] (in-process, config search).
REL_PM_TUNED: dict[tuple[int, int], dict[int, int]] = {(12, 10): {256: 2}}  # -0.19 % at 1 x 256, bit-identical
REL_PM_OVERRIDES: dict[int, int] = {}


def rel_per_core_m(device, nb: int, S: int, batch: int) -> int:
    from models.demos.embedding_gemma2.tt.audio import linear as alin

    if batch == 1:
        if S in REL_PM_OVERRIDES:
            return REL_PM_OVERRIDES[S]
        g = device.compute_with_storage_grid_size()
        if alin.AUDIO_TUNED_ON and S in REL_PM_TUNED.get((g.x, g.y), {}):
            return REL_PM_TUNED[(g.x, g.y)][S]
    return max(d for d in (1, 2, 4) if nb % d == 0)


def rel_position_table(cfg: AudioAttentionConfig) -> torch.Tensor:
    """HF `Gemma4AudioRelPositionalEncoding` output [13, hidden] in fp32: rows are distances 12 .. 0."""
    num_timescales = cfg.hidden_size // 2
    inc = math.log(10000.0 / 1.0) / max(num_timescales - 1, 1)
    inv = 1.0 * torch.exp(torch.arange(num_timescales) * -inc)
    pos = torch.arange(cfg.max_past_horizon, -1, -1)[:, None]
    scaled = pos * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).float()


def rel_key_matrix(rel_k_proj: torch.Tensor, cfg: AudioAttentionConfig) -> torch.Tensor:
    """E^T [1, H, head_dim, 32] (fp32): column c = 12 - d holds relative_k_proj(sinusoid(d)), d = 0..12
    (c = max_past_horizon - d; columns 13..31 are zero)."""
    if cfg.max_past_horizon >= REL_WIDTH:
        raise ValueError(f"max_past_horizon {cfg.max_past_horizon} must be < {REL_WIDTH}")
    relk = rel_position_table(cfg) @ rel_k_proj.float().t()  # [13, H*hd]; row p <-> distance 12 - p
    relk = relk.view(-1, cfg.num_heads, cfg.head_dim)
    E = torch.zeros(cfg.num_heads, REL_WIDTH, cfg.head_dim)
    for p in range(relk.shape[0]):
        d = cfg.max_past_horizon - p
        E[:, cfg.max_past_horizon - d] = relk[p]
    return E.transpose(-1, -2).unsqueeze(0).contiguous()


def rel_shift_index(cfg: AudioAttentionConfig) -> torch.Tensor:
    """Gather index [1, 1, 32, 64] for the relative shift: bd[i, j] = X[i, idx[i, j]].

    Query row i of a block sits at context row 32 + i, so key column j is at distance
    d = 32 + i - j and X column c = max_past_horizon - d = j - i - 32 + max_past_horizon. Off-band
    entries (d outside 0..12, masked by the attention mask) read a clamped in-range column."""
    i = torch.arange(ATTN_BLOCK)[:, None]
    j = torch.arange(CTX)[None, :]
    idx = (j - i - ATTN_BLOCK + cfg.max_past_horizon).clamp(0, REL_WIDTH - 1)
    return idx.to(torch.int32).reshape(1, 1, ATTN_BLOCK, CTX)


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
    dev = x.device()
    mc = audio_act_memory_config(dev, x.shape, x.dtype)  # E5c: L1 when it fits
    # A device-side zero block (x is finite). Not deallocating `first`: for S == 32 the slice is x itself.
    first = ttnn.slice(x, (0, 0, 0, 0), (B, H, ATTN_BLOCK, D), memory_config=mc)
    zeros = ttnn.multiply(first, 0.0, memory_config=mc)
    if nb > 1:
        head = ttnn.slice(x, (0, 0, 0, 0), (B, H, S - ATTN_BLOCK, D), memory_config=mc)
        prev = ttnn.concat([zeros, head], dim=2, memory_config=mc)
        ttnn.deallocate(head)
        ttnn.deallocate(zeros)
    else:
        prev = zeros
    prev_b = ttnn.reshape(prev, (B, H * nb, ATTN_BLOCK, D))
    own_b = ttnn.reshape(x, (B, H * nb, ATTN_BLOCK, D))
    ctx = ttnn.concat([prev_b, own_b], dim=2, memory_config=audio_act_memory_config(dev, (2, *tuple(x.shape)), x.dtype))
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
        shift_index: ttnn.Tensor,
        *,
        layer_idx: int = 0,
        precision: AudioPrecision | None = None,
    ):
        super().__init__()
        precision = precision or resolve_precision(None).audio
        self.config = config
        self.wqkv = wqkv
        self.qkv_bounds = qkv_bounds
        self.post = post
        self.qscale = qscale
        self.rel_keys_t = rel_keys_t
        self.shift_index = shift_index
        # band-attention block matmuls + softmax sum: `audio.attention` (E5c: HiFi4 + fp32 acc)
        self.compute_kernel_config = precision.attention.compute_kernel_config
        self.qkv_policy = precision.matmul("attn_qkv", layer_idx)
        self.proj_kernel_config = self.qkv_policy.compute_kernel_config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioAttentionConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "audio.self_attn",
        act_dtype=None,
        layer_idx: int = 0,
        precision: AudioPrecision | None = None,
    ) -> "AudioAttention":
        """`state_dict` holds the HF `layers.<i>.self_attn.` sub-state (`.linear.` folded). `precision`:
        the audio policy (None: the default precision file), applied as layer `layer_idx`: the fused
        QKV and `post` weights / compute configs / output dtypes (`attn_qkv`, `attn_o`; E5c: q/k/v
        fp32), the block matmuls (`attention`). `act_dtype` (not None) overrides `post`'s output dtype.

        q / k / v run as one fused projection (weights concatenated to [1024, 3072]) after one input
        clamp. That needs identical q / k / v input bounds, which holds in all 12 layers of
        google/embeddinggemma-2 (E5c `bounds.py`); other bounds raise."""
        precision = precision or resolve_precision(None).audio
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
        wqkv = as_weight(
            linear_weight(wqkv),
            device,
            cache_dir=cache_dir,
            name=f"{name}.wqkv",
            dtype=precision.matmul("attn_qkv", layer_idx).ttnn_dtype,
        )
        post = ClippableLinear.from_state_dict(
            substate(state_dict, "post"),
            H,
            H,
            device,
            name=f"{name}.post",
            out_dtype=act_dtype,
            policy=precision.matmul("attn_o", layer_idx),
            **kw,
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
            rel_key_matrix(relw, config), device, cache_dir=cache_dir, name=f"{name}.rel_keys_t32", dtype=f32
        )
        shift_index = as_weight(
            rel_shift_index(config),
            device,
            cache_dir=cache_dir,
            name=f"audio.rel_shift_index_h{config.max_past_horizon}",
            dtype=ttnn.uint16,
        )
        return cls(config, wqkv, bounds, post, qscale, rel_t, shift_index, layer_idx=layer_idx, precision=precision)

    # ------------------------------------------------------------------ helpers
    def _qkv_heads(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """x [B, 1, S, 1024] -> scaled q, scaled k, v, each [B, H, S, hd] fp32.

        q = clamp_q(x' Wq) * qscale, k = clamp_k(x' Wk) * k_scale, v = clamp_v(x' Wv) with
        x' = clamp_in(x): one fused matmul, one head split, and each output clamp applied as the
        SFPU pre-activation of its scale multiply (same ops, same order as separate clamp + mul)."""
        cfg = self.config
        dev = x.device()
        bq, bk, bv = self.qkv_bounds
        # E5c: intermediates in L1 when they fit (config.audio_act_memory_config)
        xc = ttnn.clamp(
            x, min=bq.input_min, max=bq.input_max, memory_config=audio_act_memory_config(dev, x.shape, x.dtype)
        )
        # the fp32 fused QKV output gets a wider L1 cap (72 KB per bank at 1 x 768), like the named
        # wide intermediates of the vision tower (E5b)
        # (output dtype: `audio.weights.attn_qkv.out_dtype`, fp32 in E5c)
        qkv_dtype = self.qkv_policy.out_ttnn_dtype
        qkv_shape = (x.shape[0], x.shape[2], 3 * cfg.hidden_size)
        qkv_mc = audio_act_memory_config(dev, qkv_shape, qkv_dtype, bytes_per_bank=WIDE_L1_BYTES_PER_BANK)
        qkv = audio_linear(
            xc, self.wqkv, dtype=qkv_dtype, compute_kernel_config=self.proj_kernel_config, memory_config=qkv_mc
        )
        ttnn.deallocate(xc)
        mc = audio_act_memory_config(dev, (x.shape[0], x.shape[2], cfg.hidden_size), qkv_dtype)  # one of q/k/v
        q_r, k_r, v_r = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=cfg.num_heads, num_kv_heads=cfg.num_heads, transpose_k_heads=False, memory_config=mc
        )
        ttnn.deallocate(qkv)

        def clamp_op(b):
            return ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, b.output_min, b.output_max)

        q = ttnn.multiply(q_r, self.qscale, input_tensor_a_activations=[clamp_op(bq)], memory_config=mc)
        ttnn.deallocate(q_r)
        k = ttnn.multiply(k_r, cfg.k_scale, input_tensor_a_activations=[clamp_op(bk)], memory_config=mc)
        ttnn.deallocate(k_r)
        v = ttnn.clamp(v_r, min=bv.output_min, max=bv.output_max, memory_config=mc)
        ttnn.deallocate(v_r)
        return q, k, v

    def shift_index_for(self, B: int, S: int) -> ttnn.Tensor:
        """The gather index of `_rel_shift` for a [B, 1, S, *] input: rel_shift_index repeated over the
        B x (heads x S / 32) blocks, [B, M, 32, 64] uint16. The same in every layer (the index is a
        function of max_past_horizon only), so AudioModel builds it once per forward (E10.1)."""
        M = self.config.num_heads * (S // ATTN_BLOCK)
        return ttnn.repeat(
            self.shift_index,
            ttnn.Shape([B, M, 1, 1]),
            memory_config=audio_act_memory_config(self.shift_index.device(), (B, M, ATTN_BLOCK, CTX), ttnn.bfloat16),
        )

    def _rel_shift(self, X: ttnn.Tensor, idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """X [B, M, 32, 32] -> [B, M, 32, 64] with out[i, j] = X[i, j - i - 20] (rel_shift_index): one
        tile-layout gather (E5c; was a row-major reshape / slice view trick on a 96-wide X). `idx`: the
        index from `shift_index_for` (None: built here and freed)."""
        B, M = X.shape[0], X.shape[1]
        dev = X.device()
        own = idx is None
        if own:
            idx = self.shift_index_for(B, M // self.config.num_heads * ATTN_BLOCK)
        out = ttnn.gather(
            X, dim=-1, index=idx, memory_config=audio_act_memory_config(dev, (B, M, ATTN_BLOCK, CTX), ttnn.float32)
        )
        if own:
            ttnn.deallocate(idx)
        return out

    def _softmax(self, s: ttnn.Tensor) -> ttnn.Tensor:
        dev = s.device()
        mc = audio_act_memory_config(dev, s.shape, s.dtype)
        mc_col = audio_act_memory_config(dev, (*tuple(s.shape)[:-1], ttnn.TILE_SIZE), s.dtype)
        # exp(s - max) and e * (1 / sum) with the exp / reciprocal as SFPU activations of the binary
        # ops (same ops in the same order as separate subtract / exp / reciprocal / multiply)
        m = ttnn.max(s, dim=-1, keepdim=True, memory_config=mc_col)
        exp = ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)  # 0: accurate (fast_and_approximate_mode=False)
        e = ttnn.subtract(s, m, activations=[exp], memory_config=mc)
        ttnn.deallocate(m)
        denom = ttnn.sum(
            e, dim=-1, keepdim=True, compute_kernel_config=self.compute_kernel_config, memory_config=mc_col
        )
        recip = ttnn.UnaryWithParam(ttnn.UnaryOpType.RECIP)
        p = ttnn.multiply(e, denom, input_tensor_b_activations=[recip], memory_config=mc)
        ttnn.deallocate(e)
        ttnn.deallocate(denom)
        return p

    # ------------------------------------------------------------------ forward
    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor, shift_idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """`shift_idx`: the rel-shift gather index (`shift_index_for(B, S)`), shared by the layers of one
        forward; None builds it here."""
        cfg = self.config
        ckc = self.compute_kernel_config
        B, _, S, _ = x.shape
        if S % ATTN_BLOCK != 0:
            raise ValueError(f"audio attention needs S % {ATTN_BLOCK} == 0, got {S}")
        nb, H, hd = S // ATTN_BLOCK, cfg.num_heads, cfg.head_dim
        dev = x.device()
        f32 = ttnn.float32
        # E5c: intermediates in L1 when they fit (config.audio_act_memory_config)
        mc_logits = audio_act_memory_config(dev, (B, H * nb, ATTN_BLOCK, CTX), f32)

        q, k, v = self._qkv_heads(x)

        # relative-position term on the [S, 32] layout (one column per distance), then shifted into
        # the 64-key context
        rel_t = self.rel_keys_t if B == 1 else ttnn.repeat(self.rel_keys_t, ttnn.Shape([B, 1, 1, 1]))
        pm = rel_per_core_m(dev, nb, S, B)
        X = ttnn.matmul(
            q,
            rel_t,
            program_config=block_bmm_config(dev, S, hd, REL_WIDTH, per_core_M=pm),
            compute_kernel_config=ckc,
            memory_config=audio_act_memory_config(dev, (B, H, S, REL_WIDTH), f32),
        )
        if B != 1:
            ttnn.deallocate(rel_t)
        Xb = ttnn.reshape(X, (B, H * nb, ATTN_BLOCK, REL_WIDTH))
        bd = self._rel_shift(Xb, shift_idx)
        ttnn.deallocate(X)

        qb = ttnn.reshape(q, (B, H * nb, ATTN_BLOCK, hd))
        k_ctx = block_context(k)
        v_ctx = block_context(v)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ac = ttnn.matmul(
            qb,
            k_ctx,
            transpose_b=True,
            program_config=block_bmm_config(dev, ATTN_BLOCK, hd, CTX),
            compute_kernel_config=ckc,
            memory_config=mc_logits,
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
        capped = ttnn.add(ac, bd, activations=softcap, memory_config=mc_logits)
        ttnn.deallocate(ac)
        ttnn.deallocate(bd)
        capped = ttnn.reshape(capped, (B, H, S, CTX))
        masked = ttnn.add(capped, mask, memory_config=mc_logits)
        ttnn.deallocate(capped)
        probs = self._softmax(masked)
        ttnn.deallocate(masked)
        probs = ttnn.reshape(probs, (B, H * nb, ATTN_BLOCK, CTX))

        o = ttnn.matmul(
            probs,
            v_ctx,
            program_config=block_bmm_config(dev, ATTN_BLOCK, CTX, hd),
            compute_kernel_config=ckc,
            memory_config=audio_act_memory_config(dev, (B, H, S, hd), f32),
        )  # [B, H*nb, 32, hd]
        ttnn.deallocate(probs)
        ttnn.deallocate(v_ctx)
        o = ttnn.reshape(o, (B, H, S, hd))
        o_cat = ttnn.experimental.nlp_concat_heads(
            o, memory_config=audio_act_memory_config(dev, (B, H, S, hd), f32)
        )  # [B, 1, S, H*hd]
        ttnn.deallocate(o)
        out = self.post(o_cat)
        ttnn.deallocate(o_cat)
        return out
