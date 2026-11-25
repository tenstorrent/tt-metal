# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLA + DSA prefill and decode, tensor parallel over heads (modeling_glm5_next.py:772-1276).

Absorbed NoPE MLA: ``q_a`` / ``kv_a`` (and their RMSNorms) replicated; ``q_b``, ``W_uk``, ``W_uv`` sharded by head
(``64 / tp`` heads per chip); ``o_proj`` row parallel + all-reduce. Queries attend the shared 512-wide latent as a
single KV head (K = V = latent), so the latent history is the per-sequence KV cache (replicated, grown per chunk and trimmed to the real tokens,
so a decode step -- a chunk of one real row at any position -- appends exactly one latent).

DSA: the indexer runs replicated on every chip. Its pooled keys (softmax-gated groups of ``kpool = 4`` tokens from
position 0) grow with the sequence; a query at position ``t`` selects the top ``index_topk / kpool`` complete pools
ending at or before ``t`` plus its incomplete tail ``[4 floor((t + 1) / 4), t]``. While every query of a chunk sees at
most ``index_topk / kpool`` complete pools (``t <= index_topk + kpool - 2``) the selection is all of the causal
context, so the indexer scores are skipped and plain causal attention is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from .common import DRAM, ChunkInfo, TPContext, WeightLoader, to_rm, to_tile

INDEX_ROW_BLOCK = 256  # query rows per indexer-score block


@dataclass
class MLAState:
    """Per-sequence cache of one MLA layer, trimmed to the ``length`` real tokens (ROW_MAJOR, replicated)."""

    length: int = 0
    latent: Optional[ttnn.Tensor] = None  # bf16 [1, 1, length, 512]
    pooled: Optional[ttnn.Tensor] = None  # fp32 [1, 1, length // kpool, 128] complete-pool indexer keys
    pend_k: Optional[ttnn.Tensor] = None  # fp32 [1, 1, length % kpool, 128] indexer keys of the incomplete pool
    pend_gate: Optional[ttnn.Tensor] = None  # fp32 [1, 1, length % kpool, 128] their pool gate scores


def _cat_rows(a: Optional[ttnn.Tensor], b: ttnn.Tensor) -> ttnn.Tensor:
    return b if a is None else ttnn.concat([a, b], dim=2)


def _rows(x: ttnn.Tensor, r0: int, r1: int) -> Optional[ttnn.Tensor]:
    """Rows [r0, r1) of an RM [1, 1, R, W] tensor (None when empty)."""
    if r1 <= r0:
        return None
    return ttnn.slice(x, (0, 0, r0, 0), (1, 1, r1, x.shape[-1]))


def _pad_rows_to(ctx: TPContext, x: ttnn.Tensor, rows: int) -> ttnn.Tensor:
    """RM [1, 1, R, W] -> [1, 1, rows, W] with zero rows appended."""
    r = x.shape[-2]
    return x if r == rows else ttnn.concat([x, ctx.zeros_rm(rows - r, x.shape[-1], x.dtype)], dim=2)


def _norm_gamma(t):
    return t.float().reshape(1, 1, 1, -1)


class MLAPrefill:
    def __init__(self, ctx: TPContext, cfg, src, load: WeightLoader):
        self.ctx = ctx
        tp = ctx.tp
        self.H = cfg.num_attention_heads
        assert self.H % tp == 0
        self.h = self.H // tp
        self.dqk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.dv = cfg.v_head_dim
        self.q_lora = cfg.q_lora_rank
        self.kv_lora = cfg.kv_lora_rank
        self.eps = cfg.rms_norm_eps
        self.scale = self.dqk**-0.5
        self.kpool = cfg.index_kpool
        self.topk_pools = cfg.index_topk // cfg.index_kpool
        self.n_idx = cfg.index_n_heads
        self.d_idx = cfg.index_head_dim
        sa = lambda n: src[f"self_attn.{n}"]

        def kv_b():  # [H, dqk + dv, kv_lora]
            return sa("kv_b_proj.weight").reshape(self.H, self.dqk + self.dv, self.kv_lora)

        self.w_qa_kva = load(
            "w_qa_kva",
            lambda: torch.cat([sa("q_a_proj.weight").t(), sa("kv_a_proj_with_mqa.weight").t()], 1),
            dtype=ttnn.bfloat8_b,
        )
        self.qa_norm = load("qa_norm", lambda: _norm_gamma(sa("q_a_layernorm.weight")))
        self.kva_norm = load("kva_norm", lambda: _norm_gamma(sa("kv_a_layernorm.weight")))
        self.w_qb = load("w_qb", lambda: sa("q_b_proj.weight").t(), dtype=ttnn.bfloat8_b, shard_dim=-1)
        self.w_uk = load("w_uk", lambda: kv_b()[:, : self.dqk, :][None], shard_dim=1)  # [1, H, dqk, 512]
        self.w_uv = load("w_uv", lambda: kv_b()[:, self.dqk :, :].transpose(1, 2)[None], shard_dim=1)  # [1, H, 512, dv]
        self.w_o = load("w_o", lambda: sa("o_proj.weight").t(), dtype=ttnn.bfloat8_b, shard_dim=0)

        ix = lambda n: src[f"self_attn.indexer.{n}"]
        self.w_idx_x = load(
            "idx.w_x",
            lambda: torch.cat(
                [ix("wk.weight").t(), ix("index_kpool_compress_gate").t(), ix("weights_proj.weight").t()], 1
            ),
        )  # [D, 128 + 128 + 32]: key, pool gate, head weights
        self.w_idx_q = load("idx.wq_b", lambda: ix("wq_b.weight").t())
        self.k_norm_w = load("idx.k_norm_w", lambda: _norm_gamma(ix("k_norm.weight")), dtype=ttnn.float32)
        self.k_norm_b = load("idx.k_norm_b", lambda: _norm_gamma(ix("k_norm.bias")), dtype=ttnn.float32)
        self.ape = [
            load(
                f"idx.ape{j}",
                (lambda j=j: ix("index_kpool_compress_ape")[j].float().reshape(1, 1, 1, -1)),
                dtype=ttnn.float32,
            )
            for j in range(self.kpool)
        ]

    # ---------------------------------------------------------------------------------------------------------
    def _pool(self, k_rm, gate_rm, n):
        """``n`` complete pools from RM fp32 [1, 1, kpool n, 128] keys / gate scores -> pooled keys RM fp32
        [1, 1, n, 128] (softmax over each group's gate + ape, rounded to bf16 as HF keeps them)."""
        P, d = self.kpool, self.d_idx
        group = lambda x: to_tile(ttnn.reshape(x, (1, 1, n, P * d)))
        kg, gg = group(k_rm), group(gate_rm)
        cols = lambda x, j: ttnn.slice(x, (0, 0, 0, j * d), (1, 1, n, (j + 1) * d))
        logits = [ttnn.add(cols(gg, j), self.ape[j]) for j in range(P)]
        m = logits[0]
        for lg in logits[1:]:
            m = ttnn.maximum(m, lg)
        e = [ttnn.exp(ttnn.subtract(lg, m)) for lg in logits]
        den = e[0]
        for x in e[1:]:
            den = ttnn.add(den, x)
        num = ttnn.multiply(e[0], cols(kg, 0))
        for j in range(1, P):
            num = ttnn.addcmul(num, e[j], cols(kg, j))
        pooled = ttnn.typecast(ttnn.typecast(ttnn.div(num, den), ttnn.bfloat16), ttnn.float32)
        return to_rm(pooled, ttnn.float32)

    def _layer_norm(self, x):
        mu = ttnn.mean(x, dim=-1, keepdim=True)
        xc = ttnn.subtract(x, mu)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
        y = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, 1e-6)))
        return ttnn.add(ttnn.multiply(y, self.k_norm_w), self.k_norm_b)

    def _indexer_cache(self, x, state: MLAState, info: ChunkInfo):
        """This chunk's indexer keys / gates appended to the incomplete pool -> (all complete pooled keys RM fp32,
        new incomplete-pool keys, gates, head weights of the chunk rows [1, 1, t, 32] fp32)."""
        ctx = self.ctx
        t, d = info.t, self.d_idx
        bf = lambda u: ttnn.typecast(ttnn.typecast(u, ttnn.bfloat16), ttnn.float32)  # HF's bf16 module outputs
        ix = ttnn.matmul(x, self.w_idx_x, dtype=ttnn.float32, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        k_idx = bf(self._layer_norm(bf(ttnn.slice(ix, (0, 0, 0, 0), (1, 1, t, d)))))
        gate = bf(ttnn.slice(ix, (0, 0, 0, d), (1, 1, t, 2 * d)))
        wts = ttnn.slice(ix, (0, 0, 0, 2 * d), (1, 1, t, 2 * d + self.n_idx))
        k_ext = _cat_rows(state.pend_k, to_rm(k_idx, ttnn.float32))
        g_ext = _cat_rows(state.pend_gate, to_rm(gate, ttnn.float32))
        real = (state.pend_k.shape[-2] if state.pend_k is not None else 0) + info.n_valid
        n_new = real // self.kpool
        pooled = state.pooled
        if n_new:
            rows = n_new * self.kpool
            pooled = _cat_rows(pooled, self._pool(_rows(k_ext, 0, rows), _rows(g_ext, 0, rows), n_new))
        done = n_new * self.kpool
        return pooled, _rows(k_ext, done, real), _rows(g_ext, done, real), wts

    def _dsa_mask(self, q_r, pooled, info: ChunkInfo, wts):
        """Additive bf16 [1, 1, t, kv_pad] mask of the indexer's selection for every query row."""
        ctx = self.ctx
        t, kv_pad, width = info.t, info.kv_pad, info.pool_width
        qi = ttnn.matmul(q_r, self.w_idx_q, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        qi = ttnn.permute(ttnn.reshape(qi, (1, t, self.n_idx, self.d_idx)), (0, 2, 1, 3))  # [1, 32, t, 128]
        pk_t = ttnn.transpose(to_tile(_pad_rows_to(ctx, pooled, width)), -2, -1)  # [1, 1, 128, width]
        wts = ttnn.permute(ttnn.multiply(wts, self.n_idx**-0.5), (0, 3, 2, 1))  # [1, 32, t, 1]
        pool_valid, tail = info.pool_valid(), info.tail()
        masks = []
        for r0 in range(0, t, INDEX_ROW_BLOCK):
            r1 = min(t, r0 + INDEX_ROW_BLOCK)
            rows = r1 - r0
            qb = ttnn.slice(qi, (0, 0, r0, 0), (1, self.n_idx, r1, self.d_idx))
            sc = ttnn.matmul(qb, pk_t, dtype=ttnn.float32, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
            sc = ttnn.relu(ttnn.multiply(sc, self.d_idx**-0.5))
            sc = ttnn.multiply(sc, ttnn.slice(wts, (0, 0, r0, 0), (1, self.n_idx, r1, 1)))
            score = ttnn.sum(sc, dim=1, keepdim=True)  # [1, 1, rows, width]
            ttnn.deallocate(sc)
            pv = ttnn.slice(pool_valid, (0, 0, r0, 0), (1, 1, r1, width))
            score = ttnn.where(pv, score, -1e30)
            if width > self.topk_pools:
                vals, idx = ttnn.topk(score, k=self.topk_pools, dim=-1, largest=True, sorted=True)
                ttnn.deallocate(idx)
                kth = ttnn.slice(vals, (0, 0, 0, self.topk_pools - 1), (1, 1, rows, self.topk_pools))
                sel = ttnn.multiply(ttnn.ge(score, kth), pv)
            else:
                sel = pv
            sel = ttnn.repeat_interleave(ttnn.typecast(sel, ttnn.bfloat16), self.kpool, dim=3)  # [1, 1, rows, 4 width]
            if self.kpool * width >= kv_pad:
                sel = ttnn.slice(sel, (0, 0, 0, 0), (1, 1, rows, kv_pad))
            else:
                pad = ttnn.zeros(
                    (1, 1, rows, kv_pad - self.kpool * width),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=ctx.device,
                    memory_config=DRAM,
                )
                sel = ttnn.concat([sel, pad], dim=3)
            vis = ttnn.add(sel, ttnn.slice(tail, (0, 0, r0, 0), (1, 1, r1, kv_pad)))
            masks.append(ttnn.where(ttnn.gtz(vis), 0.0, ChunkInfo.NEG))
        mask = masks[0] if len(masks) == 1 else ttnn.concat(masks, dim=2)
        return ttnn.typecast(mask, ttnn.bfloat16)

    # ---------------------------------------------------------------------------------------------------------
    def forward(self, x: ttnn.Tensor, state: MLAState, info: ChunkInfo):
        """x: normalized sublayer input bf16 TILE [1, 1, t, D] of the chunk at ``info.start == state.length`` ->
        (bf16 TILE [1, 1, t, D] replicated, new state holding the cache of the ``start + n_valid`` real tokens)."""
        ctx = self.ctx
        t = x.shape[-2]
        assert info.start == state.length, f"chunk starts at {info.start}, cache holds {state.length} tokens"
        qk = ttnn.matmul(x, self.w_qa_kva, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.qa_norm, epsilon=self.eps, compute_kernel_config=ctx.hifi4)
        c = ttnn.rms_norm(c, weight=self.kva_norm, epsilon=self.eps, compute_kernel_config=ctx.hifi4)
        latent_rm = _cat_rows(state.latent, to_rm(c))  # [1, 1, start + t, 512]: cached tokens + this chunk
        latent = to_tile(_pad_rows_to(ctx, latent_rm, info.kv_pad))

        pooled, pend_k, pend_gate, wts = self._indexer_cache(x, state, info)

        q = ttnn.matmul(q_r, self.w_qb, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        q = ttnn.permute(ttnn.reshape(q, (1, t, self.h, self.dqk)), (0, 2, 1, 3))  # [1, h, t, dqk]
        q_lat = ttnn.matmul(q, self.w_uk, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        ttnn.deallocate(q)

        sparse = info.last_pos > self.topk_pools * self.kpool + self.kpool - 2
        if sparse:
            mask = self._dsa_mask(q_r, pooled, info, wts)
        elif info.start > 0 or info.kv_pad != t:
            mask = info.causal()
        else:
            mask = None
        chunk = 128 if info.aligned else 32
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ctx.device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            q_lat,
            latent,
            latent,
            attn_mask=mask,
            is_causal=mask is None,
            scale=self.scale,
            program_config=pc,
            compute_kernel_config=ctx.hifi4,
            memory_config=DRAM,
        )
        ttnn.deallocate(q_lat)
        ttnn.deallocate(latent)
        if sparse:
            ttnn.deallocate(mask)
        o = ttnn.matmul(o, self.w_uv, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, t, self.h * self.dv))
        y = ttnn.matmul(o, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        length = info.start + info.n_valid
        new_state = MLAState(
            length=length,
            latent=_rows(latent_rm, 0, length),
            pooled=pooled,
            pend_k=pend_k,
            pend_gate=pend_gate,
        )
        return ctx.all_reduce(y), new_state
