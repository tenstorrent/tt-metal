# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 DSA indexer (dsa_moe indexer step) on the 2x2 mesh: top 512 pools of 4 keys + the query's tail tokens.

Keys (all S rows, every chip): k = LayerNorm(x wk^T) (k_norm w + b, eps 1e-6); gate = x g^T + ape[t % 4]; pooled key
p = sum_j softmax_j(gate[4p + j]) k[4p + j] (fp32, elementwise over 4 slices of the [S/4, 512] reshape), written into
the replicated pooled-key cache [1, 1, max_seq / 4 + 32, 128] bf16 at row start / 4 (ttnn.fill_cache).
Queries (chip d = 2 r + c takes rows d S/4 .. (d + 1) S/4, two ttnn.mesh_partition): q = heads(q_resid wq_b^T),
w = x weights_proj^T / 64 (32^-0.5 128^-0.5 folded, exact) -> ttnn.experimental.indexer_score_dsa over the whole cache
with chunk_start_idx = kv_len = (start + S) / 4 (every pool below kv_len visible to every row; the op needs a tile-aligned
start below T, so the cache has 32 spare rows) -> the chunk's own pool columns get a constant pool-causal mask
(pool j visible iff 4 j + 3 <= row) -> ttnn.experimental.topk_large_indices (k 512) over the first kv_len columns.
Token ids (fp32, exact below 2^24): [4 p | 4 p + 1 | 4 p + 2 | 4 p + 3] (column order is free), then a 128-wide block
of tail tokens (constant relative table + start) padded with -1; rows before position 2047 (only at start 0) take the
constant dense causal row 0..q. Output: [1, 1, S/4, 2176] uint32 row-major per chip, 0xFFFFFFFF sentinels as a
contiguous tail (sparse_sdpa's index format). Every per-chunk constant is built at load for each chunk size.
Reuse: models/demos/deepseek_v3_d_p/tt/mla/indexer.py (score / top-k plumbing), without its RoPE / block-cyclic paths.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate

KP = 4
TOPK_POOLS = 512
OUT_W = 2176  # 2048 pool tokens + 128 (3 tail + sentinel pad): sparse_sdpa wants TOPK % 128 == 0
TAIL_W = OUT_W - KP * TOPK_POOLS
DENSE_END = KP * TOPK_POOLS - 1  # positions below this select every token 0..q
MC = ttnn.DRAM_MEMORY_CONFIG
# "heads": per-head ttnn.linear + fp32 addcmul; "op": ttnn.experimental.indexer_score_dsa (bf16 head sum)
SCORE_MODE = os.environ.get("GLM_INDEXER_SCORE", "heads")


class TtIndexer:
    def __init__(self, mesh, w: dict, cfg, max_seq: int, chunks):
        self.mesh = mesh
        self.nh, self.hd = cfg.index_n_heads, cfg.index_head_dim
        assert cfg.index_kpool == KP and cfg.index_topk == KP * TOPK_POOLS
        self.ndev = mesh.get_num_devices()
        assert tuple(mesh.shape) == (2, 2), "query split assumes a 2x2 mesh (chip d = 2 r + c)"
        self.score_mode = SCORE_MODE
        self.mm = hifi4_config()
        self.score_cfg = hifi4_config(fp32_acc=False)  # the score op honours only math_fidelity
        up = lambda t: replicate(
            mesh, t.float().T.reshape(1, 1, t.shape[1], t.shape[0]).to(torch.bfloat16)
        )  # noqa: E731
        self.wk = up(w["wk"])
        self.wg = up(w["gate"])
        self.wq = up(w["wq_b"])
        self.wp = up(w["weights_proj"].float() * (self.nh**-0.5 * self.hd**-0.5))
        self.kn_w = replicate(mesh, w["k_norm_w"].float().reshape(1, 1, 1, -1).to(torch.bfloat16))
        self.kn_b = replicate(mesh, w["k_norm_b"].float().reshape(1, 1, 1, -1).to(torch.bfloat16))
        self.ape = replicate(mesh, w["ape"].float().reshape(1, 1, 1, KP * self.hd), dtype=ttnn.float32)
        self.cache_rows = max_seq // KP + ttnn.TILE_SIZE
        self.cache = pooled_key_cache(mesh, cfg, max_seq)
        self.consts = {s: self._chunk_consts(s) for s in sorted(set(chunks))}

    # ---- load-time constants, one set per chunk size (each chip holds its own S/4 rows)
    def _sharded(self, t: torch.Tensor, dtype) -> ttnn.Tensor:
        """[S, W] host rows in chip order d -> chip (r, c) holds rows (2 r + c) S/4 .. ."""
        s, wdt = t.shape
        return ttnn.from_torch(
            t.reshape(2, 2, s // self.ndev, wdt).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=MC,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, 1), mesh_shape=tuple(self.mesh.shape)),
        )

    def _chunk_consts(self, s: int) -> dict:
        assert s % (self.ndev * ttnn.TILE_SIZE) == 0 and s // KP % ttnn.TILE_SIZE == 0, f"chunk {s} not tile aligned"
        assert s // KP >= TOPK_POOLS, f"chunk {s}: the first chunk's {s // KP} pools are fewer than top-k {TOPK_POOLS}"
        r = torch.arange(s)[:, None]  # position relative to the chunk start
        j = torch.arange(s // KP)[None, :]
        pool_mask = torch.where(KP * j + KP - 1 <= r, 0.0, float("-inf"))  # [S, S/4] own pools
        tail_n = (r + 1) % KP
        cols = torch.arange(TAIL_W)[None, :]
        valid = cols < tail_n
        rel = torch.where(valid, r + 1 - tail_n + cols, torch.full_like(cols, -1)).float()
        c = torch.arange(OUT_W)[None, :]
        dense = torch.where(c <= r, c, torch.full_like(c, -1)).float()  # used only at start 0
        dense_sel = (r < DENSE_END).float()
        return {
            "pool_mask": self._sharded(pool_mask, ttnn.bfloat16),
            "tail_rel": self._sharded(rel, ttnn.float32),
            "tail_valid": self._sharded(valid.float(), ttnn.float32),
            "dense": self._sharded(dense, ttnn.float32),
            "dense_sel": self._sharded(dense_sel, ttnn.float32),
        }

    # ---- forward
    def _pooled_keys(self, x: ttnn.Tensor, s: int) -> ttnn.Tensor:
        """x [1, 1, S, H] -> pooled keys [1, 1, S/4, 128] bf16."""
        m, hd = s // KP, self.hd
        k = ttnn.linear(x, self.wk, dtype=ttnn.float32, compute_kernel_config=self.mm, memory_config=MC)
        kn = ttnn.layer_norm(
            k, weight=self.kn_w, bias=self.kn_b, epsilon=1e-6, compute_kernel_config=self.mm, memory_config=MC
        )
        ttnn.deallocate(k)
        g = ttnn.linear(x, self.wg, dtype=ttnn.float32, compute_kernel_config=self.mm, memory_config=MC)
        g4 = ttnn.reshape(g, (1, 1, m, KP * hd))
        k4 = ttnn.reshape(kn, (1, 1, m, KP * hd))
        lg = ttnn.add(g4, self.ape, memory_config=MC)
        del g, g4, kn  # reshape may alias its input: let the last reference free the buffer
        sl = lambda t, i: ttnn.slice(t, (0, 0, 0, i * hd), (1, 1, m, (i + 1) * hd), memory_config=MC)  # noqa: E731
        ls = [sl(lg, i) for i in range(KP)]
        ks = [sl(k4, i) for i in range(KP)]
        del lg, k4
        mx = ttnn.maximum(ttnn.maximum(ls[0], ls[1]), ttnn.maximum(ls[2], ls[3]))
        es = [ttnn.exp(ttnn.subtract(li, mx, memory_config=MC), memory_config=MC) for li in ls]
        den = ttnn.add(ttnn.add(es[0], es[1]), ttnn.add(es[2], es[3]))
        num = ttnn.multiply(es[0], ks[0])
        for i in range(1, KP):
            num = ttnn.add(num, ttnn.multiply(es[i], ks[i]))
        pooled = ttnn.div(num, den, memory_config=MC)
        for t in ls + ks + es + [mx, den, num]:
            ttnn.deallocate(t)
        out = ttnn.typecast(pooled, ttnn.bfloat16, memory_config=MC)
        ttnn.deallocate(pooled)
        return out

    def _local_rows(self, t: ttnn.Tensor) -> ttnn.Tensor:
        a = ttnn.mesh_partition(t, dim=-2, cluster_axis=0, memory_config=MC)
        b = ttnn.mesh_partition(a, dim=-2, cluster_axis=1, memory_config=MC)
        ttnn.deallocate(a)
        return b

    def __call__(self, x: ttnn.Tensor, q_resid: ttnn.Tensor, start: int, q_local: bool = False) -> ttnn.Tensor:
        """x (attn_norm) [1, 1, S, H], q_resid [1, 1, S, 1536], both replicated bf16 TILE; start a multiple of 4.
        q_local: q_resid is already this chip's [1, 1, S/4, 1536] rows (the split residual layout).
        Returns this chip's [1, 1, S/4, 2176] uint32 ROW_MAJOR token ids (0xFFFFFFFF = none). Updates the cache."""
        s = x.shape[-2]
        assert start % KP == 0 and start % s == 0, f"chunk start {start} must be a multiple of S={s}"
        c = self.consts[s]
        sq, p0 = s // self.ndev, start // KP
        kv = p0 + s // KP
        assert kv + ttnn.TILE_SIZE <= self.cache_rows, f"chunk end {start + s} past max_seq"

        pooled = self._pooled_keys(x, s)
        ttnn.fill_cache(self.cache, pooled, batch_idx=0, update_idx=p0)
        ttnn.deallocate(pooled)

        qr = q_resid if q_local else self._local_rows(q_resid)
        q = ttnn.linear(qr, self.wq, dtype=ttnn.bfloat16, compute_kernel_config=self.mm, memory_config=MC)
        if qr is not q_resid:
            ttnn.deallocate(qr)
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=MC
        )
        ttnn.deallocate(q)
        xl = self._local_rows(x)
        wdt = ttnn.float32 if self.score_mode == "heads" else ttnn.bfloat16
        wts = ttnn.linear(xl, self.wp, dtype=wdt, compute_kernel_config=self.mm, memory_config=MC)
        ttnn.deallocate(xl)
        score = self._scores_heads(qh, wts, kv) if self.score_mode == "heads" else self._scores_op(qh, wts, kv)
        ttnn.deallocate(qh)
        ttnn.deallocate(wts)
        # own pools get the pool-causal mask; earlier pools are visible to every row
        own = ttnn.slice(score, (0, 0, 0, p0), (1, 1, sq, kv), memory_config=MC)
        own_m = ttnn.add(own, c["pool_mask"], memory_config=MC)
        ttnn.deallocate(own)
        if p0:
            pre = ttnn.slice(score, (0, 0, 0, 0), (1, 1, sq, p0), memory_config=MC)
            ranked_t = ttnn.concat([pre, own_m], dim=-1, memory_config=MC)
            ttnn.deallocate(pre)
            ttnn.deallocate(own_m)
        else:
            ranked_t = own_m
        ttnn.deallocate(score)
        ranked = ttnn.to_layout(ranked_t, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        ttnn.deallocate(ranked_t)
        ids = ttnn.experimental.topk_large_indices(ranked, k=TOPK_POOLS)  # [1, 1, S/4, 512] uint32, pool ids
        ttnn.deallocate(ranked)

        idf = ttnn.typecast(ttnn.to_layout(ids, ttnn.TILE_LAYOUT), ttnn.float32, memory_config=MC)
        ttnn.deallocate(ids)
        b4 = ttnn.multiply(idf, float(KP), memory_config=MC)
        ttnn.deallocate(idf)
        toks = [b4] + [ttnn.add(b4, float(i), memory_config=MC) for i in range(1, KP)]
        tail = ttnn.add(ttnn.multiply(c["tail_valid"], float(start), memory_config=MC), c["tail_rel"], memory_config=MC)
        out = ttnn.concat(toks + [tail], dim=-1, memory_config=MC)  # [1, 1, S/4, 2176] fp32
        for t in toks + [tail]:
            ttnn.deallocate(t)
        if start < DENSE_END:
            dense = ttnn.where(c["dense_sel"], c["dense"], out)
            ttnn.deallocate(out)
            out = dense
        oi = ttnn.typecast(out, ttnn.int32, memory_config=MC)
        ttnn.deallocate(out)
        orm = ttnn.to_layout(oi, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        ttnn.deallocate(oi)
        return ttnn.bitcast(orm, ttnn.uint32)

    def _scores_op(self, qh, wts, kv):
        """indexer_score_dsa over the whole cache, unmasked below kv -> [1, 1, S/4, kv] bf16 TILE. The op packs each
        head's relu(q.k) to bf16 and sums the 32 heads in a bf16 DEST."""
        sq = qh.shape[-2]
        sc = ttnn.experimental.indexer_score_dsa(
            qh, self.cache, wts, chunk_start_idx=kv, compute_kernel_config=self.score_cfg
        )
        v = ttnn.slice(sc, (0, 0, 0, 0), (1, 1, sq, kv), memory_config=MC)
        ttnn.deallocate(sc)
        out = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=MC)
        ttnn.deallocate(v)
        return out

    def _scores_heads(self, qh, wts, kv):
        """sum_h relu(q_h k^T) w_h per head: ttnn.linear (HiFi4, fp32 acc, relu) + addcmul (fp32 column broadcast)
        -> [1, 1, S/4, kv] bf16 TILE (topk_large_indices ranks bf16)."""
        sq = qh.shape[-2]
        k = ttnn.slice(self.cache, (0, 0, 0, 0), (1, 1, kv, self.hd), memory_config=MC)
        kt = ttnn.transpose(k, -2, -1, memory_config=MC)  # [1, 1, 128, kv]
        ttnn.deallocate(k)
        acc = None
        for h in range(self.nh):
            q1 = ttnn.slice(qh, (0, h, 0, 0), (1, h + 1, sq, self.hd), memory_config=MC)
            m = ttnn.linear(
                q1, kt, dtype=ttnn.float32, activation="relu", compute_kernel_config=self.mm, memory_config=MC
            )
            ttnn.deallocate(q1)
            w1 = ttnn.slice(wts, (0, 0, 0, h), (1, 1, sq, h + 1), memory_config=MC)
            if acc is None:
                acc = ttnn.multiply(m, w1, memory_config=MC)
            else:
                nxt = ttnn.addcmul(acc, m, w1, memory_config=MC)
                ttnn.deallocate(acc)
                acc = nxt
            ttnn.deallocate(m)
            ttnn.deallocate(w1)
        ttnn.deallocate(kt)
        out = ttnn.typecast(acc, ttnn.bfloat16, memory_config=MC)
        ttnn.deallocate(acc)
        return out

    # ---- state at the harness boundary (prefix load / read-back; never inside the forward)
    def bind_cache(self, cache: ttnn.Tensor) -> None:
        """Read and write another pooled-key cache of the same shape (pooled_key_cache; one per serving slot)."""
        assert tuple(cache.shape) == tuple(self.cache.shape), (cache.shape, self.cache.shape)
        self.cache = cache

    def load_state(self, tensors: dict, length: int | None = None) -> None:
        """Pooled keys [n, 128] (reference layout, rows [0, length / 4) valid) -> the device cache."""
        pk = tensors["index_key"].float()
        n = pk.shape[0] if length is None else length // KP
        host = torch.zeros(1, 1, self.cache_rows, self.hd)
        host[0, 0, :n] = pk[:n]
        d = replicate(self.mesh, host.to(torch.bfloat16))
        ttnn.copy(d, self.cache)
        ttnn.deallocate(d)

    def state_torch(self) -> dict:
        """The pooled-key cache [max_seq / 4 + 32, 128] (chip 0's copy; replicated)."""
        return {"index_key": ttnn.to_torch(ttnn.get_device_tensors(self.cache)[0]).reshape(self.cache_rows, -1).float()}


def pooled_key_cache(mesh, cfg, max_seq: int) -> ttnn.Tensor:
    """The replicated pooled-key cache [1, 1, max_seq / 4 + 32, 128] bf16 TILE, zeroed (32 spare rows, see above)."""
    return ttnn.zeros(
        (1, 1, max_seq // KP + ttnn.TILE_SIZE, cfg.index_head_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=MC,
    )


def build_indexer(mesh, loader, cfg, layer: int, max_seq: int, chunks) -> TtIndexer:
    p = f"{PREFIX}layers.{layer}.self_attn.indexer."
    w = {
        "wk": loader.weight(p + "wk.weight"),
        "gate": loader.weight(p + "index_kpool_compress_gate"),
        "ape": loader.weight(p + "index_kpool_compress_ape"),
        "wq_b": loader.weight(p + "wq_b.weight"),
        "weights_proj": loader.weight(p + "weights_proj.weight"),
        "k_norm_w": loader.weight(p + "k_norm.weight"),
        "k_norm_b": loader.weight(p + "k_norm.bias"),
    }
    return TtIndexer(mesh, w, cfg, max_seq, chunks)
