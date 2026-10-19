# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash DSA indexer with k-pool compression (``Glm5NextTextIndexer``), single device.

HF semantics (index_kpool = 4, index_topk = 2048, always_select_tail):
    q  = wq_b(q_resid)                         [T, 32, 128]
    k  = LayerNorm(wk(x))                      [T, 128]
    g  = x @ gate^T                            [T, 128]      (k-pool gate logits)
    pool p (tokens 4p..4p+3, complete pools only):
        pk[p] = sum_j softmax_j(g[4p+j] + ape[j]) * k[4p+j]              [128]
    score[s, p] = sum_h relu(q[s,h] . pk[p] * 128^-0.5) * w[s,h] * 32^-0.5, w = weights_proj(x)
    pool p is selectable by query s iff 4p+3 <= s; top-512 pools -> their 4 tokens (2048 ids)
    + the incomplete tail tokens 4*floor((s+1)/4) .. s (<= 3 ids), -1 elsewhere.
For s < 2048 every visible token is selected and the attention is exact (the dense path).

Device composite (design 5.3 / 8.3):
  * pooled keys live in a per-layer cache ``pool_cache [max_batch*pool_blocks, 1, 32, 128]`` bf16
    TILE (paged view, identity page table; user u's pools are contiguous) and are also readable as
    ``[max_batch, 1, pools_per_user, 128]`` for ``indexer_score_dsa(cache_batch_idx=u)``;
  * the raw ``k`` and gate logits of the last 3 tokens are kept per user in a 4-slot ring
    (slot = position % 4) so decode can finish the pool that the new token completes;
  * scoring: ``ttnn.experimental.indexer_score_dsa(key_compression_ratio=4)`` (causal by pool);
  * selection: the pool scores are repeated 4x along the key axis and
    ``ttnn.experimental.topk_large_indices(k=2048)`` returns TOKEN ids directly (ties inside a
    pool are contiguous, so whole pools are taken), with 0xFFFFFFFF for masked keys;
  * the tail ids are built from the query positions and appended; the row is padded to a
    multiple of 128 (sparse_sdpa's k_chunk) with the 0xFFFFFFFF sentinel.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.tt.config import GLM53FlashConfig

TILE = 32
SENTINEL = 0xFFFFFFFF


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _compute(device, fidelity=ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


class TtIndexCache:
    """Per-layer pooled-key cache + raw tail rings for ``max_batch`` users."""

    def __init__(self, device, cfg: GLM53FlashConfig, *, max_batch: int, max_seq_len: int):
        self.device = device
        self.D = cfg.index_head_dim
        self.P = cfg.index_kpool
        self.max_batch = max_batch
        self.pools_per_user = -(-max_seq_len // self.P)
        self.pools_per_user = -(-self.pools_per_user // TILE) * TILE  # tile-aligned rows per user
        self.pool_blocks = self.pools_per_user // TILE
        # paged view for row writes; the same buffer viewed [B,1,pools,D] for scoring
        self.pool_cache = ttnn.zeros(
            (max_batch * self.pool_blocks, 1, TILE, self.D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt = torch.arange(max_batch * self.pool_blocks, dtype=torch.int32).reshape(max_batch, self.pool_blocks)
        self.page_table = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # raw tail rings: k and gate logits of the last tokens, slot = position % P
        self.tail_k = ttnn.zeros(
            (max_batch, 1, self.P, self.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.tail_g = ttnn.zeros(
            (max_batch, 1, self.P, self.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def write_pools(self, pooled: ttnn.Tensor, user: int):
        """pooled [1,1,n_pad,D] bf16 (n_pad % 32 == 0) -> pools [0, n_pad) of ``user``."""
        ttnn.experimental.paged_fill_cache(self.pool_cache, pooled, self.page_table, batch_idx=user)

    def update_pool(self, pooled_sharded: ttnn.Tensor, pool_ids: ttnn.Tensor):
        """pooled [1,B,1(32),D] height-sharded on B cores; pool_ids [B] int32 (row to write per user)."""
        ttnn.experimental.paged_update_cache(
            self.pool_cache, pooled_sharded, update_idxs_tensor=pool_ids, page_table=self.page_table
        )


class TtDSAIndexer(LightweightModule):
    def __init__(self, device, cfg: GLM53FlashConfig, idx_state_dict: dict[str, torch.Tensor]):
        """``idx_state_dict``: raw tensors of ``self_attn.indexer.`` (prefix stripped)."""
        self.device = device
        self.cfg = cfg
        self.Hi, self.D, self.P = cfg.index_n_heads, cfg.index_head_dim, cfg.index_kpool
        sd = idx_state_dict
        # softmax scale folded into wq_b, the n_heads^-0.5 into weights_proj (relu is positively homogeneous)
        self.w_qb = _upload(
            device, (sd["wq_b.weight"].float() * cfg.index_softmax_scale).t(), ttnn.bfloat16
        )  # [1536, Hi*D]
        self.w_k = _upload(device, sd["wk.weight"].t(), ttnn.bfloat16)  # [4096, D]
        self.knorm_w = _upload(device, sd["k_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.knorm_b = _upload(device, sd["k_norm.bias"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_weights = _upload(
            device, (sd["weights_proj.weight"].float() * self.Hi**-0.5).t(), ttnn.bfloat16
        )  # [4096, Hi]
        self.w_gate = _upload(device, sd["index_kpool_compress_gate"].t(), ttnn.bfloat16)  # [4096, D]
        self.ape = _upload(
            device, sd["index_kpool_compress_ape"].float().reshape(1, 1, self.P, self.D), ttnn.float32
        )  # [1,1,P,D]
        self.hifi2 = _compute(device, ttnn.MathFidelity.HiFi2, packer_l1_acc=True)
        self.fp32 = _compute(device)
        self.score_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)

    # ------------------------------------------------------------- stems
    def keys_and_gates(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """x [1,1,T,4096] bf16 -> (k [1,1,T,D] bf16 (LayerNorm'd), g [1,1,T,D] bf16 gate logits)."""
        k = ttnn.linear(x, self.w_k, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        k = ttnn.layer_norm(k, weight=self.knorm_w, bias=self.knorm_b, epsilon=1e-6, compute_kernel_config=self.fp32)
        g = ttnn.linear(x, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return k, g

    def pool(self, k: ttnn.Tensor, g: ttnn.Tensor) -> ttnn.Tensor:
        """k, g [1,1,n*P,D] (complete pools only) -> pooled keys [1,1,n,D] bf16."""
        n = k.shape[-2] // self.P
        kp = ttnn.reshape(ttnn.typecast(k, ttnn.float32), [1, n, self.P, self.D])
        gp = ttnn.reshape(ttnn.typecast(g, ttnn.float32), [1, n, self.P, self.D])
        logits = ttnn.add(gp, self.ape)  # [1,n,P,D] (+ ape[j] per in-pool position)
        prob = ttnn.softmax(logits, dim=-2, compute_kernel_config=self.fp32)  # over the P tokens of each pool
        pooled = ttnn.sum(ttnn.multiply(prob, kp), dim=-2, keepdim=True)  # [1,n,1,D]
        return ttnn.typecast(ttnn.reshape(pooled, [1, 1, n, self.D]), ttnn.bfloat16)

    def queries_and_weights(self, x: ttnn.Tensor, q_resid: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """-> q [1,Hi,T,D] bf16 (scale folded), w [1,1,T,Hi] bf16 (n_heads^-0.5 folded)."""
        T = x.shape[-2]
        q = ttnn.linear(q_resid, self.w_qb, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,Hi*D]
        q = ttnn.transpose(ttnn.reshape(q, [1, T, self.Hi, self.D]), 1, 2)  # [1,Hi,T,D]
        w = ttnn.linear(x, self.w_weights, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,Hi]
        return q, w

    # ---------------------------------------------------------- selection
    def select(
        self,
        q: ttnn.Tensor,
        w: ttnn.Tensor,
        cache: TtIndexCache,
        user: int,
        *,
        query_start: int,
        num_pools_valid: int,
    ) -> ttnn.Tensor:
        """Top-2048 token ids (+ tail, sentinel-padded to a multiple of 128) for queries at absolute
        positions ``query_start + s``. Returns [1,1,T,W] uint32 ROW_MAJOR."""
        T = q.shape[-2]
        # one spare tile beyond the valid pools: the op requires the (32-aligned) chunk start to lie inside
        # kv_len*P tokens, and a decode query at position a may sit in a block starting at a - a%32 >= 4*pools.
        # The spare rows are causally invisible to the real query (pool p visible iff 4p+3 <= a).
        kv_len = min(cache.pools_per_user, max(TILE, -(-(num_pools_valid + 1) // TILE) * TILE))
        # batched [B,1,pools,D] view of the paged pool cache, taken after the latest writes
        pools_batched = ttnn.reshape(cache.pool_cache, [cache.max_batch, 1, cache.pools_per_user, cache.D])
        scores = ttnn.experimental.indexer_score_dsa(
            q,
            pools_batched,
            w,
            chunk_start_idx=query_start,
            key_compression_ratio=self.P,
            program_config=self.score_cfg,
            cache_batch_idx=user,
            kv_len=kv_len,
        )  # [1,1,T,pools_per_user] bf16, -inf where not visible
        scores_rm = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        expanded = ttnn.repeat_interleave(scores_rm, self.P, dim=-1)  # [1,1,T,4*pools] token-level scores
        top = ttnn.experimental.topk_large_indices(expanded, k=self.cfg.index_topk, valid_length=kv_len * self.P)
        # [1,1,T,2048] uint32 token ids, 0xFFFFFFFF where -inf. (to_layout / repeat_interleave may alias
        # their inputs, so the intermediates are released together once the ids exist.)
        for t in (scores, scores_rm, expanded):
            if t.is_allocated():
                ttnn.deallocate(t)
        tail = self._tail_ids(T, query_start)  # [1,1,T,P-1] uint32
        width = self.cfg.index_topk + self.P - 1
        pad = (-width) % 128
        parts = [top, tail]
        if pad:
            parts.append(self._sentinels(T, pad))
        idx = ttnn.concat(parts, dim=-1)
        ttnn.deallocate(top)
        ttnn.deallocate(tail)
        return idx

    def _tail_ids(self, T: int, query_start: int) -> ttnn.Tensor:
        """Token ids of the incomplete tail pool of each query (position a = query_start + s):
        4*floor((a+1)/4) + j for j < (a+1) % 4, else the sentinel. int32 arithmetic on device
        with -1 as the sentinel (bit-identical to 0xFFFFFFFF after the uint32 reinterpretation)."""
        P = self.P
        rows = torch.arange(T, dtype=torch.int32).reshape(1, 1, T, 1) + query_start  # a
        base = (rows + 1) // P * P
        ids = torch.stack([base + j for j in range(P - 1)], dim=-1).reshape(1, 1, T, P - 1)
        valid = ids <= rows
        ids = torch.where(valid, ids.to(torch.int64), torch.full_like(ids, SENTINEL, dtype=torch.int64))
        # host-built from the (static) chunk geometry: query_start and T are program shape, not data
        return ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def _sentinels(self, T: int, n: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.full((1, 1, T, n), SENTINEL, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
