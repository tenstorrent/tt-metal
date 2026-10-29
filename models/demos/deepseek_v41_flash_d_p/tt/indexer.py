# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 indexer of an index-source layer with its own compressed index keys (reference/ds41_ref.py:_indexer,
enc_source layers 2 / 8 / 14: ratio 2; the candidate source layer 20: ratio 1 plus the candidate pool,
TtCandidateIndexer below; the re-index layers 24 / 28 / 32 / 36 over layer 20's keys and pool, TtReindexer) on the
SP x TP mesh. Adapted from models/demos/hy4_preview_d_p/tt/indexer.py
(ring score over SP, TP query sub-shard, topk_large_indices, TP gather) and deepseek_v3_d_p/tt/mla/indexer.py:
TtCsaIndexer (compressed keys); no Hadamard (the DS41 reference has none), keys from the compressor's served cache.

    q     = fp4_qdq(rope(fp8_linear(q_resid, wq_b)), 32)      [S, 32 heads, 128], RoPE on each head's last 64 dims
    w     = (attn_norm @ weights_proj^T) / 64                  [S, 32] (fp32 x, bf16 weight, no act quant)
    score = sum_h w_h relu(q_h . k_t), key t visible iff t < (pos + 1) // ratio
    topk  = the 512 best visible keys per query (fewer visible: pads)
(the released code's fp8 / fp4 quant-dequant of the query and its matmul input is not emulated: the device keeps
bf16 and computes in it, agent rule 17; FIX.8. INDEX_Q_FORMAT / INDEX_K_FORMAT=bfp4 (DS41-150, the defaults): the
query and the keys in the device's own 4-bit format, bfloat4_b, rounded to nearest even by ttnn.bringup.bfp_round;
the score op and its ring read them as such)

Per chip (r, c), inputs attn_norm [1, 1, W, H/TP] fp32 (hidden split) and q_resid [1, 1, W, 1280] fp32 (row split,
replicated over TP), W = S / SP; the chip's queries are rows [c Sq, (c+1) Sq) of its row block, Sq = W / TP, i.e.
positions start + r W + c Sq + [0, Sq) (natural blocks, any 32-aligned start):

    w   attn_norm (fp32) @ the chip's K rows of (weights_proj / 64)^T (bf16, exact: 1/64 = 2^-6) -> tt/common.tp_sum (exact
        fp32, K-30) -> the chip's Sq rows (mesh_partition over axis 1) -> bf16 (the score op's gate dtype);
        INDEXER_GLUE=on (P.16): the linear writes the CCL's partial dtype and ONE reduce-scatter by row blocks
        (tt/ccl.sum_rows) gives the chip's Sq rows of the sum directly
    q   the chip's Sq rows of q_resid -> bf16 @ wq_b^T (replicated, all 32 heads; DENSE_WEIGHTS; every head's rope
        rows de-interleaved at load, tt/rope.deinterleave_perm) at ATTN_FIDELITY + fp32 acc -> [1, 32, Sq, 128] fp32
        -> tt/rope.rope_heads (fp32 SFPU, the reference's interleaved order back) -> bf16; ROPE_IMPL=on (P.17, the
        default): wq_b in checkpoint order writing bf16 -> nlp_create_q_heads_split -> ONE stock rotary op with the
        nope prepended (tt/rope.rope_stock)
    k   the layer's served index-key cache (tt/compressor.TtServedRows, 2_index_k_r2: SP-sharded, TP-replicated,
        bf16 ROW_MAJOR in token units, [rows, 64] per chip; the compressor wrote this chunk's keys) viewed as
        compressed rows [rows / 2, 128] and tiled: block-cyclic over SP with W / 2 compressed rows per SP rank per
        chunk (position-based: token g on mesh row (g / W) % SP)
    score ttnn.bringup.ring_indexer_score_dsa (fork indexer_score: fp32 DEST + HiFi4, k_chunk 32, K-13 / K-14),
        ringing the keys over axis 0, key_compression_ratio 2 (fork option: causal t < (pos + 1) // 2) and
        natural_query_blocks (fork option: queries start + r W + c Sq + i at any tile-aligned start while the keys
        keep the cache's block-cyclic layout) -> [1, 1, Sq, T] bf16 ROW_MAJOR, columns >= kv_len unwritten
    top-k ttnn.experimental.topk_large_indices(k 512, valid_length = kv_len) -> [1, 1, Sq, 512] uint32 (descending
        score, -inf -> 0xFFFFFFFF tail) -> all_gather over axis 1 -> [1, 1, W, 512], replicated over TP;
        INDEXER_TOPK_LOCAL=on (P.8, ``local``, set by the all-device model): no gather, the chip keeps its own
        [1, 1, Sq, 512] rows (the sparse attention reads exactly those)

ATTN_SP32=on (P.10, ``full_rows``; tt/model.py): attn_norm [1, 1, Sq, H] and q_resid [1, 1, Sq, q_lora] are already
the chip's query rows: w = ONE local matmul with weights_proj replicated (whole K in fp32 DEST, written bf16; no TP
sum), q_resid read as it is (no mesh_partition).

The key buffer the ring gathers into ([1, 1, T, 128], T = the cache's compressed capacity) is allocated once per
cache capacity (``setup``). No host work in ``__call__``: the per-chunk inputs are integers (start) and device
tensors. Ties: the scores are bf16 (topk_large_indices is bf16-only), so near ties flip where the reference's fp32
scores do not (plan.md "Accuracy notes", K-42).
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from . import mm_configs as mm
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS, attn_compute_config, cast, dense_dtype, gemm_input, tp_sum
from .rope import RopeTable, deinterleave_perm, indexer_glue, rope_heads, rope_stock
from .settings import settings

Q_CHUNK, K_CHUNK = 64, 32  # score work unit: k 32 is the fork's fidelity-honouring head reduction (K-13 / K-14)


def lowbit_host() -> bool:
    """LOWBIT_HOST=on (P.12): the low-bit writes' host time cut (bfp_round compact into persistent buffers here; the
    window blocks in one compact call in tt/attention.py)."""
    return settings.get("LOWBIT_HOST") == "on"


class Kept:
    """A persistent tensor held by a ChunkMemo slot (tt/sparse_attention.py): the memo frees ttnn.Tensor values when
    it drops a slot, not this wrapper's."""

    __slots__ = ("t",)

    def __init__(self, t: ttnn.Tensor):
        self.t = t


def topk_local() -> bool:
    """INDEXER_TOPK_LOCAL=on (P.8): the all-device model's indexers keep each chip's own top-k rows (no TP gather)."""
    return settings.get("INDEXER_TOPK_LOCAL") == "on"


def _upload(mesh, t, dtype, layout=ttnn.TILE_LAYOUT, dims=None):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=(
            ttnn.ReplicateTensorToMesh(mesh)
            if dims is None
            else ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)
        ),
    )


class TtIndexer:
    """indexer step of one index-source layer that owns its index keys (enc_source). ``setup(cache)`` once per
    served index-key cache capacity (load time), then ``__call__(x, qr, start, cache)`` per chunk: attn_norm
    [1, 1, W, H/TP] fp32 + q_resid [1, 1, W, q_lora] fp32 -> top-k [1, 1, W, topk] uint32 ROW_MAJOR (key
    positions; 0xFFFFFFFF = none), row-split, replicated over TP; with ``local`` (INDEXER_TOPK_LOCAL=on, P.8: the
    all-device model sets it; component harnesses keep the gathered layout) [1, 1, Sq, topk], the chip's own query
    rows. Does not consume its inputs."""

    def __init__(
        self, mesh, w: dict[str, torch.Tensor], cfg, ratio: int, rope: RopeTable, chunk: int, full_rows: bool = False
    ):
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

        # ATTN_SP32=on (P.10, ``full_rows``): attn_norm [1, 1, Sq, H] and q_resid [1, 1, Sq, q_lora] are the chip's own
        # query rows: weights_proj replicated, the gates one local matmul (no TP sum), q_resid used as it is
        self.full_rows = full_rows

        self.mesh, self.rope, self.ratio, self.chunk = mesh, rope, ratio, chunk
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.nh, self.hd, self.rd, self.topk = cfg.index_heads, cfg.index_dim, cfg.rope_dim, cfg.index_topk
        assert ratio in (2, 4) or ratio == 1, ratio
        assert self.sp > 1, "the ring score needs an SP axis of at least 2 chips"
        self.w_rows = chunk // self.sp  # W: query tokens per SP row per chunk
        assert self.w_rows % (32 * self.tp) == 0, (chunk, self.sp, self.tp)
        self.sq = self.w_rows // self.tp  # query rows per chip
        assert (self.w_rows // ratio) % 32 == 0, f"W / ratio = {self.w_rows // ratio} compressed rows not tile-aligned"
        assert chunk // ratio >= self.topk, f"a chunk's {chunk // ratio} keys < topk {self.topk} (valid_length >= k)"
        assert rope.half * 2 == self.rd, (rope.half, self.rd)
        self.ckc = attn_compute_config(mesh)  # weights_proj, wq_b: ATTN_FIDELITY (P.7)
        # wq_b [nh * hd, q_lora] -> W^T [q_lora, nh * hd], every head's rope rows de-interleaved; replicated
        # ROPE_IMPL=on (P.17): checkpoint order (the stock rope reads interleaved pairs)
        self.stock = rope.stock
        perm = torch.arange(self.hd) if self.stock else deinterleave_perm(self.hd, self.rd)
        wq = w["wq_b"].float()
        q_lora = wq.shape[1]
        wq = wq.view(self.nh, self.hd, q_lora)[:, perm, :].reshape(self.nh * self.hd, q_lora)
        wq_t = wq.t().contiguous()
        assert torch.equal(wq_t.bfloat16().float(), wq_t), "wq_b must be bf16-exact (e4m3 x 2^k)"
        # DENSE_WEIGHTS (P.13): bf16 (exact fp8 values) or bfloat8_b (the checkpoint's 8-bit width). weights_proj is
        # bf16 in the checkpoint: it stays bf16 (rule 7)
        self.wq_b = _upload(mesh, wq_t.reshape(1, 1, q_lora, -1), dense_dtype(wq_t, "indexer.wq_b"))
        # weights_proj [nh, H] scaled by hd^-0.5 nh^-0.5 (a power of two: exact) -> W^T [H, nh], K rows split over TP
        scale = self.hd**-0.5 * self.nh**-0.5
        wp = w["weights_proj"].float() * scale
        assert torch.equal(wp.bfloat16().float(), wp), "weights_proj x scale must stay bf16-exact"
        hidden = wp.shape[1]
        assert hidden % (32 * self.tp) == 0, (hidden, self.tp)
        dims = (None, 2) if TP_AXIS == 1 else (2, None)
        if full_rows:
            dims = None  # replicated
        self.wproj = _upload(mesh, wp.t().reshape(1, 1, hidden, self.nh), ttnn.bfloat16, dims=dims)
        self.ccl = get_tt_ccl(mesh)
        self.topology = ttnn.Topology.Linear  # FABRIC_2D: no wrap assumed on the SP axis (every box)
        self.num_links = 2 if mesh.arch() == ttnn.Arch.BLACKHOLE else 1
        # the q work unit divides the chip's query rows (indexer_score: whole q chunks): 64 at the seed's 1280 rows,
        # 32 at 8x4's 160 (5 tiles; DS41-117)
        q_chunk = max(q for q in range(32, min(Q_CHUNK, self.sq) + 1, 32) if self.sq % q == 0)
        self.score_cfg = ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=q_chunk, k_chunk_size=K_CHUNK, head_group_size=0
        )
        self.score_ckc = attn_compute_config(mesh)  # the ring score: ATTN_FIDELITY (P.7)
        self._kbuf = {}  # compressed capacity T -> the ring's persistent gathered-key buffer [1, 1, T, hd]
        self.glue = indexer_glue()  # INDEXER_GLUE=on (P.16)
        self._memo = None  # (tt/sparse_attention.ChunkMemo, KV source layer): ``share``
        self.local = False  # INDEXER_TOPK_LOCAL=on (P.8): the top-k not gathered over TP (tt/model.py sets it)
        # INDEX_K_FORMAT / INDEX_Q_FORMAT=bfp4 (DS41-150): keys and query rounded to bfloat4_b before the score
        self.k_dtype = ttnn.bfloat4_b if settings.get("INDEX_K_FORMAT") == "bfp4" else ttnn.bfloat16
        self.q_bfp4 = settings.get("INDEX_Q_FORMAT") == "bfp4"
        self.prepared = settings.get("CKV_BFP4_PREPARED") == "on"  # P.7: bfp_round built once, addresses patched
        # LOWBIT_HOST=on (P.12): bfp_round compact (no per-core runtime args) into this layer's own output buffers
        self.lowbit_host = lowbit_host()
        self._rounded = {}  # (what, shape) -> the persistent bfp_round output (LOWBIT_HOST=on)

    def share(self, memo, kv_src: int | None) -> None:
        """INDEXER_GLUE=on (P.16): the chunk's ChunkMemo and this layer's KV source (the tiled index keys are made once
        per (source, start, cache version)); ``share(None, None)``: every call tiles its own."""
        self._memo = (memo, kv_src) if memo is not None else None

    # ------------------------------------------------------------------ load time
    def setup(self, cache) -> None:
        """The ring's gathered-key buffer for this cache's capacity (allocated once per capacity)."""
        t = self._capacity(cache)
        if t not in self._kbuf:
            self._kbuf[t] = ttnn.from_torch(
                torch.zeros(1, 1, t, self.hd),
                dtype=self.k_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def _capacity(self, cache) -> int:
        assert cache.ratio == self.ratio and cache.width == self.hd and cache.tp_axis is None, (
            cache.ratio,
            cache.width,
            cache.tp_axis,
        )
        return cache.rows // self.ratio * self.sp

    # ------------------------------------------------------------------ per chunk (device only)
    def _round(self, x: ttnn.Tensor, what: str) -> tuple[ttnn.Tensor, bool]:
        """x bf16 TILE -> (x on the bfloat4_b grid, nearest even; whether that tensor is this layer's persistent
        buffer, which the caller never frees). LOWBIT_HOST=on (P.12): bfp_round compact into the layer's buffer for
        ``what`` (allocated at the first call; each call rewrites all of it, and the layer's next call of the same kind
        comes after every reader of the last one on the queue); off: a new tensor each call (FIX.14's path)."""
        dtype = self.k_dtype if what == "keys" else ttnn.bfloat4_b
        if not self.lowbit_host:
            return ttnn.bringup.bfp_round(x, dtype, memory_config=DRAM, prepared=self.prepared), False
        key = (what, tuple(x.shape))
        out = self._rounded.get(key)
        if out is None:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), dtype, ttnn.TILE_LAYOUT, self.mesh, DRAM)
            self._rounded[key] = out
        return ttnn.bringup.bfp_round(x, out=out, prepared=self.prepared, compact=True), True

    def _keys(self, cache) -> ttnn.Tensor:
        """The served index-key cache's slab [1, 1, rows / r, hd] bf16 ROW_MAJOR (ND-sharded) -> compressed rows
        [1, 1, rows / r, hd] bf16 TILE (INDEX_K_FORMAT=bfp4: rounded to bfloat4_b), interleaved: the ring's k_local
        (block-cyclic layout unchanged)."""
        rm = cache.slab.slice_rows(0, cache.crows)
        kt = ttnn.to_layout(rm, ttnn.TILE_LAYOUT, memory_config=DRAM)
        dealloc(rm)
        if self.k_dtype == ttnn.bfloat16:
            return kt
        k4, _ = self._round(kt, "keys")
        dealloc(kt)
        return k4

    def _sub(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """This chip's Sq query rows of a row-split, TP-replicated [1, 1, W, n] tensor (ATTN_SP32=on, P.10: ``t``
        itself, already the chip's rows; the caller frees the result only when it is not ``t``)."""
        if self.full_rows and t.shape[-2] == self.sq:
            return t
        if self.tp == 1:
            return ttnn.clone(t, memory_config=DRAM)
        return ttnn.mesh_partition(t, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)

    def gates(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """attn_norm [1, 1, W, H/TP] fp32 -> the chip's per-head gates [1, 1, Sq, nh] bf16."""
        # INDEXER_GLUE=on (P.16): the partial in the CCL's partial dtype (bf16 under fabric_bf16, as q_a's) and ONE
        # reduce-scatter by row blocks: its block is the chip's Sq query rows (no all_gather, mesh_partition, casts)
        if self.full_rows:  # ATTN_SP32=on (P.10): the chip's rows, the whole sum in fp32 DEST, written bf16
            return mm.linear(
                x,
                self.wproj,
                site="indexer.weights_proj",
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.ckc,
                memory_config=DRAM,
            )
        rs = self.glue and settings.get("HC_TP_SUM") == "gather_add"
        pdt = ccl.partial_dtype(self.mesh) if rs else ttnn.float32
        part = mm.linear(
            x, self.wproj, site="indexer.weights_proj", dtype=pdt, compute_kernel_config=self.ckc, memory_config=DRAM
        )
        sub = ccl.sum_rows(self.mesh, part) if rs else None
        if sub is not None:
            dealloc(part)
            if sub.dtype == ttnn.bfloat16:
                return sub
            out = ttnn.typecast(sub, ttnn.bfloat16, memory_config=DRAM)
            dealloc(sub)
            return out
        if part.dtype != ttnn.float32:  # (tp_sum's stock / untiled path sums fp32)
            pf = ttnn.typecast(part, ttnn.float32, memory_config=DRAM)
            dealloc(part)
            part = pf
        full = tp_sum(self.mesh, part)
        dealloc(part)
        sub = self._sub(full)
        dealloc(full)
        out = ttnn.typecast(sub, ttnn.bfloat16, memory_config=DRAM)
        dealloc(sub)
        return out

    def query(self, qr: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """q_resid [1, 1, W, q_lora] fp32 -> the chip's index queries [1, nh, Sq, hd] bf16."""
        sub = self._sub(qr)
        qq = gemm_input(sub)  # GEMM_INPUTS (FIX.7): bfloat8_b, the reference's fp8 act quant of q_resid
        if qq is not sub and sub is not qr:
            dealloc(sub)
        qdt = ttnn.bfloat16 if self.stock else ttnn.float32  # ROPE_IMPL=on: the bf16 heads the stock rope reads
        q = mm.linear(qq, self.wq_b, site="indexer.wq_b", dtype=qdt, compute_kernel_config=self.ckc, memory_config=DRAM)
        if qq is not qr:  # (ATTN_SP32=on: qr itself when it is already the matmul's dtype)
            dealloc(qq)
        if self.stock:
            return self._query_stock(q, start)
        if self.rope.pairs is not None:
            return self._query_glue(q, start)
        q4 = ttnn.reshape(q, [1, self.sq, self.nh, self.hd])
        qh = ttnn.permute(q4, (0, 2, 1, 3), memory_config=DRAM)  # [1, nh, Sq, hd]
        dealloc(q)
        cos, sin = self.rope.rows(start, self.chunk, tp_split=True)  # the chip's Sq positions
        qr_ = rope_heads(qh, cos, sin, interleave=True)
        dealloc(qh)
        self.rope.free(cos, sin)  # INDEXER_GLUE=on: the chunk's memoized rows stay
        out = cast(qr_, ttnn.bfloat16)
        if out is not qr_:
            dealloc(qr_)
        return out

    def _query_stock(self, q: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """ROPE_IMPL=on (P.17): wq_b's bf16 output [1, 1, Sq, nh hd] (consumed) -> the index queries [1, nh, Sq, hd]
        bf16: the heads split by nlp_create_q_heads_split, then ONE stock rotary op on the [.., 64] tails with the
        nope prepended (tt/rope.rope_stock, concat_prefix)."""
        qn, qt = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.nh, split_head_dim=self.hd - self.rd, memory_config=DRAM
        )
        dealloc(q)
        cos, sin = self.rope.rows(start, self.chunk, tp_split=True)  # the chip's Sq positions
        out = rope_stock(qt, cos, sin, self.rope.trans, prefix=qn)
        self.rope.free(cos, sin)
        dealloc(qt)
        dealloc(qn)
        return out

    def _query_glue(self, q: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """ATTN_GLUE=on (P.15): wq_b's output [1, 1, Sq, nh hd] fp32 (consumed) -> the index queries [1, nh, Sq, hd]
        bf16, the values of the reshape + permute path bit for bit: the heads split by nlp_create_q_heads_split (a
        data move), RoPE on the [.., 64] tails only, its last ops writing bf16, the interleave a PairPerm matmul on the
        bf16 tail."""
        qn, qt = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.nh, split_head_dim=self.hd - self.rd, memory_config=DRAM
        )
        dealloc(q)
        cos, sin = self.rope.rows(start, self.chunk, tp_split=True)  # the chip's Sq positions
        tb = rope_heads(qt, cos, sin, dtype=ttnn.bfloat16)  # [re | ro] bf16
        dealloc(qt)
        self.rope.free(cos, sin)
        ti = self.rope.pairs(tb, self.rope.pairs.to_inter)
        dealloc(tb)
        nb = ttnn.typecast(qn, ttnn.bfloat16, memory_config=DRAM)
        dealloc(qn)
        out = ttnn.concat([nb, ti], dim=-1, memory_config=DRAM)
        dealloc(nb)
        dealloc(ti)
        return out

    def scores(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache, **epilogue) -> tuple[ttnn.Tensor, int]:
        """The chip's index scores [1, 1, Sq, T] bf16 ROW_MAJOR (future columns -inf, columns >= kv_len unwritten)
        and kv_len (the written width, a multiple of 32). ``epilogue`` (INDEXER_GLUE=on, P.16): the score op's block
        epilogue options (block_keys, block_bias, block_max_out, block_max_pin), passed through."""
        assert start % 32 == 0, start
        t = self._capacity(cache)
        kbuf = self._kbuf[t]  # setup(cache) at load
        end = start + self.chunk  # the padded window's end; later keys are causally masked for every real query
        kv_len = min(t, -(-(end // self.ratio) // 32) * 32)
        assert kv_len >= self.topk and start < kv_len * self.ratio, (start, kv_len, self.topk)
        w = self.gates(x)
        q = self.query(qr, start)
        q_kept = False  # (LOWBIT_HOST=on: q is the layer's persistent bfp_round buffer, never freed)
        if self.q_bfp4:  # INDEX_Q_FORMAT=bfp4: the roped query on the bfloat4_b grid (nearest even)
            q4, q_kept = self._round(q, "query")
            dealloc(q)
            q = q4
        shared = self._memo is not None
        kkey = (start, id(cache), cache.version)
        k_local = self._memo[0].get(("keys", self._memo[1]), kkey) if shared else None
        if isinstance(k_local, Kept):
            k_local = k_local.t
        # (LOWBIT_HOST=on with INDEX_K_FORMAT=bfp4: the keys are the layer's persistent buffer, never freed; a memo
        # holds it as Kept, which its drop leaves alone)
        k_kept = self.lowbit_host and self.k_dtype != ttnn.bfloat16
        if k_local is None:
            k_local = self._keys(cache)
            if shared:  # INDEXER_GLUE=on (P.16): the re-index layers score the same tiled keys (L24 = L20's)
                self._memo[0].put(("keys", self._memo[1]), kkey, Kept(k_local) if k_kept else k_local)
        logits = ttnn.bringup.ring_indexer_score_dsa(
            q,
            kbuf,
            w,
            k_local,
            self.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            cluster_axis=SP_AXIS,
            topology=self.topology,
            num_links=self.num_links,
            chunk_start_idx=start,
            key_compression_ratio=self.ratio,
            program_config=self.score_cfg,
            compute_kernel_config=self.score_ckc,
            kv_len=kv_len,
            seq_subshard_axis=TP_AXIS if self.tp > 1 else None,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=self.w_rows,
            natural_query_blocks=True,
            **epilogue,
        )  # [1, 1, Sq, T] bf16 ROW_MAJOR; future columns -inf, columns >= kv_len unwritten
        for u in (() if q_kept else (q,)) + (w,) + (() if shared or k_kept else (k_local,)):
            dealloc(u)
        return logits, kv_len

    def topk_of(self, logits: ttnn.Tensor, kv_len: int) -> ttnn.Tensor:
        """The 512 best keys per query of the chip's scores -> [1, 1, W, topk] uint32, gathered over TP (``local``:
        the chip's own [1, 1, Sq, topk])."""
        local = ttnn.experimental.topk_large_indices(logits, k=self.topk, valid_length=kv_len)
        if self.tp == 1 or self.local:
            return local
        out = ccl.all_gather(local, 2, TP_AXIS)  # [1, 1, W, topk]
        dealloc(local)
        return out

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> ttnn.Tensor:
        logits, kv_len = self.scores(x, qr, start, cache)
        out = self.topk_of(logits, kv_len)
        dealloc(logits)
        return out


def _same(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    """Whether two tensors share a buffer (a no-op slice / to_layout returns a new wrapper over its input's, K-1)."""
    return a.buffer_address() == b.buffer_address()


BIG = 1.0e30  # the pinned block's score boost: above any real score, finite (0 x BIG = 0 off the pin), bf16-exact


class TtCandidateIndexer(TtIndexer):
    """indexer step of the candidate-source layer (dec_candidate, layer 20: ratio 1, its own index keys). As TtIndexer
    (top 512 over every visible key: layer 20 itself is not restricted to candidates), plus the candidate pool
    (reference/ds41_ref.py:select_candidate_blocks) from the same scores, per chip:

        block score  max over each block of ``block`` (8) key positions: the scores sliced to kv_len, padded with -inf
                     to a multiple of 32 blocks, 8 strided slices (step 8) folded by ttnn.maximum (exact; bf16)
        pin          the block of the query's own (newest) position, (start + r W + c Sq + i) // 8, gets + BIG: fp32
                     integer ops on a load-time row-position column and block-column iota (exact below 2^24), eq,
                     x BIG, added (-inf + 0 stays -inf: entirely future blocks stay out)
        top blocks   topk_large_indices(k = candidate_topk_blocks (2048), the block row padded with -inf to k) ->
                     [1, 1, Sq, 2048] uint32 block ids, descending score, 0xFFFFFFFF = none (-inf: no visible key)

    ``forward`` returns (top-k [1, 1, W, 512] uint32 gathered over TP, candidates [1, 1, Sq, 2048] uint32 of the chip's
    own query rows, not gathered: the dec_reindex layers 24-36 score the same rows of the same chunk). The order
    within a row is free (index sets; ties: bf16 scores, K-42).

    INDEXER_GLUE=on (P.16): the block max and the pin come out of the score op itself (ttnn.bringup
    ring_indexer_score_dsa's block epilogue: block_keys 8, block_max_out = a load-time [1, 1, Sq, ceil32(T / 8)] fp32
    buffer, block_max_pin = +inf at each query's own block), then a bf16 cast (exact) and the same top blocks: the
    same candidate sets as the slices + maximum + pin ops."""

    def __init__(self, mesh, w, cfg, ratio: int, rope: RopeTable, chunk: int, full_rows: bool = False):
        super().__init__(mesh, w, cfg, ratio, rope, chunk, full_rows=full_rows)
        assert ratio == 1, "the candidate pool is over key positions (the candidate source is ratio 1)"
        self.block, self.ctopk = cfg.cand_block, cfg.cand_topk_blocks
        assert 16 <= self.ctopk <= 2048 and self.ctopk % 16 == 0, self.ctopk  # topk_large_indices' k range
        assert 32 % self.block == 0, self.block  # kv_len (a multiple of 32) holds whole blocks
        # position offset of each of the chip's query rows within the chunk: r W + c Sq + i (fp32, exact)
        sp, tp, sq = self.sp, self.tp, self.sq
        rows = (torch.arange(sp).view(sp, 1, 1) * self.w_rows + torch.arange(tp).view(1, tp, 1) * sq) + torch.arange(
            sq
        ).view(1, 1, sq)
        dims = (0, 1) if SP_AXIS == 0 else (1, 0)
        self.row_pos = _upload(mesh, rows.float().reshape(sp, tp, sq, 1), ttnn.float32, dims=dims)  # [1, 1, Sq, 1]
        self._cols = {}  # capacity T -> block-column iota [1, 1, 1, ceil32(T / block)] fp32
        self._bmax = {}  # INDEXER_GLUE=on: capacity T -> the score op's block-max buffer [1, 1, Sq, ceil32(T / 8)] fp32

    def setup(self, cache) -> None:
        super().setup(cache)
        t = self._capacity(cache)
        nb = -(-(t // self.block) // 32) * 32
        if self.glue:
            if t not in self._bmax:  # written by every call's score op (block_max_out), read by the top blocks
                self._bmax[t] = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, self.sq, nb]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM
                )
            return
        if t not in self._cols:
            self._cols[t] = _upload(self.mesh, torch.arange(nb, dtype=torch.float32).reshape(1, 1, 1, nb), ttnn.float32)

    def _candidates_glue(self, kv_len: int, t: int) -> ttnn.Tensor:
        """INDEXER_GLUE=on (P.16): the candidate pool from the block maxes the score op wrote (its block epilogue:
        max over each block's 8 bf16 scores as fp32, +inf at each query's own block; the old path's pin was +1e30
        in bf16: the same selection) -> bf16 (exact) -> top ctopk blocks of the first kv_len / 8."""
        nbv = kv_len // self.block
        bm = self._bmax[t]
        if nbv >= self.ctopk:
            rm = ttnn.typecast(bm, ttnn.bfloat16, memory_config=DRAM)
            out = ttnn.experimental.topk_large_indices(rm, k=self.ctopk, valid_length=nbv)
            dealloc(rm)
            return out
        sl = ttnn.slice(bm, [0, 0, 0, 0], [1, 1, self.sq, nbv], memory_config=DRAM)  # fewer blocks than k: pad -inf
        rm = ttnn.typecast(sl, ttnn.bfloat16, memory_config=DRAM)
        if not _same(sl, bm):  # DS41-52: a full-range slice is a wrapper over the persistent buffer
            dealloc(sl)
        padded = ttnn.pad(rm, [(0, 0), (0, 0), (0, 0), (0, self.ctopk - nbv)], value=float("-inf"))
        dealloc(rm)
        out = ttnn.experimental.topk_large_indices(padded, k=self.ctopk)
        dealloc(padded)
        return out

    def candidates_of(self, logits: ttnn.Tensor, kv_len: int, start: int, t: int) -> ttnn.Tensor:
        """The chip's candidate blocks [1, 1, Sq, ctopk] uint32 from its scores (does not consume them)."""
        b = self.block
        nb = -(-(kv_len // b) // 32) * 32  # block columns, tile-aligned
        sq = self.sq
        lg = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, sq, kv_len], memory_config=DRAM)  # logits itself if kv_len == T
        if nb * b > kv_len:
            padded = ttnn.pad(lg, [(0, 0), (0, 0), (0, 0), (0, nb * b - kv_len)], value=float("-inf"))
            if not _same(lg, logits):  # K-1: a full-range slice returns its input's buffer
                dealloc(lg)
            lg = padded
        acc = None
        for i in range(b):  # block max: column j of slice i = score j b + i
            part = ttnn.slice(lg, [0, 0, 0, i], [1, 1, sq, nb * b], [1, 1, 1, b], memory_config=DRAM)
            if acc is None:
                acc = part
                continue
            nxt = ttnn.maximum(acc, part, memory_config=DRAM)
            dealloc(acc)
            dealloc(part)
            acc = nxt
        if not _same(lg, logits):
            dealloc(lg)
        blk = ttnn.to_layout(acc, ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, 1, Sq, nb] bf16
        if not _same(blk, acc):  # K-1
            dealloc(acc)
        # pin: block (start + row offset) // b of every query row
        pos = ttnn.add(self.row_pos, float(start), memory_config=DRAM)
        pin = ttnn.floor(ttnn.multiply(pos, 1.0 / b, memory_config=DRAM), memory_config=DRAM)  # b a power of two
        dealloc(pos)
        cols = ttnn.slice(self._cols[t], [0, 0, 0, 0], [1, 1, 1, nb], memory_config=DRAM)
        hit = ttnn.eq(cols, pin, dtype=ttnn.float32, memory_config=DRAM)  # [1, 1, Sq, nb] 1 at the pinned block
        dealloc(pin)
        if not _same(cols, self._cols[t]):  # K-1: never free the load-time iota
            dealloc(cols)
        boost = ttnn.multiply(hit, BIG, dtype=ttnn.bfloat16, memory_config=DRAM)
        dealloc(hit)
        pinned = ttnn.add(blk, boost, dtype=ttnn.bfloat16, memory_config=DRAM)
        for u in (blk, boost):
            dealloc(u)
        rm = ttnn.to_layout(pinned, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if not _same(rm, pinned):
            dealloc(pinned)
        if nb < self.ctopk:  # fewer blocks than k: the rest are none (-inf -> 0xFFFFFFFF)
            padded = ttnn.pad(rm, [(0, 0), (0, 0), (0, 0), (0, self.ctopk - nb)], value=float("-inf"))
            dealloc(rm)
            rm = padded
        out = ttnn.experimental.topk_large_indices(rm, k=self.ctopk)
        dealloc(rm)
        return out

    def forward(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if self.glue:  # INDEXER_GLUE=on (P.16): the block maxes come out of the score op (fork block epilogue)
            t = self._capacity(cache)
            ep = dict(block_keys=self.block, block_max_out=self._bmax[t], block_max_pin=True)
            logits, kv_len = self.scores(x, qr, start, cache, **ep)
            cands = self._candidates_glue(kv_len, t)
            out = self.topk_of(logits, kv_len)
            dealloc(logits)
            return out, cands
        logits, kv_len = self.scores(x, qr, start, cache)
        cands = self.candidates_of(logits, kv_len, start, self._capacity(cache))
        out = self.topk_of(logits, kv_len)
        dealloc(logits)
        return out, cands

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> ttnn.Tensor:
        out, cands = self.forward(x, qr, start, cache)
        dealloc(cands)
        return out


class TtReindexer(TtIndexer):
    """indexer step of a decoder re-index layer (dec_reindex: 24, 28, 32, 36): its own queries and gates over the KV
    source's (layer 20's) index keys at ratio 1, restricted to the candidate source's pool of the same queries
    (reference/ds41_ref.py:_indexer, ``uses_candidates``: scores outside the query's candidate blocks -> -inf, then the
    512 best). Per chip, the same Sq query rows as the candidate source (natural blocks, any 32-aligned start):

        score    TtIndexer.scores on the KV source's served 4_index_k_r1 cache (ring over axis 0, ratio 1)
        bias     ttnn.scatter of the chip's candidate block ids [1, 1, Sq, 2048] uint32 ROW_MAJOR (layer 20's device
                 pool, any order; 0xFFFFFFFF = none, skipped by scatter as out of range) as 0 into a load-time
                 -inf row [1, 1, Sq, nb] bf16 ROW_MAJOR (bf16: K-4) -> ttnn.repeat_interleave x block (8) ->
                 [1, 1, Sq, kv_len]: 0 on every key of a candidate block, -inf elsewhere (exact)
        mask     scores [:, :kv_len] + bias (ROW_MAJOR bf16: s + 0 = s, s + -inf = -inf; future keys stay -inf)
        top-k    topk_large_indices(k 512) -> all_gather over axis 1 -> [1, 1, W, 512] uint32 (-inf -> 0xFFFFFFFF)

    A query sees at least min(pos + 1, 2047 x 8 + 1) candidate keys, so -inf never fills the 512 where the
    reference would count a masked key (its ``idx < lens`` keeps them) except below 512 visible keys, where both pad.

    INDEXER_GLUE=on (P.16): the bias (the same scatter, then tiled) goes into the score op (ring_indexer_score_dsa's
    block epilogue: block_keys 8, block_bias): its writer writes -inf over every key of a non-candidate block, so the
    scores come out masked (no repeat_interleave, slice or add)."""

    def __init__(self, mesh, w, cfg, ratio: int, rope: RopeTable, chunk: int, full_rows: bool = False):
        super().__init__(mesh, w, cfg, ratio, rope, chunk, full_rows=full_rows)
        assert ratio == 1, "the re-index layers score the candidate source's key positions (ratio 1)"
        self.block, self.ctopk = cfg.cand_block, cfg.cand_topk_blocks
        assert 32 % self.block == 0, self.block  # kv_len (a multiple of 32) holds whole blocks
        self.src = _upload(mesh, torch.zeros(1, 1, self.sq, self.ctopk), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._neg = {}  # capacity T -> the -inf block row [1, 1, Sq, ceil16(T / block)] bf16 ROW_MAJOR

    def setup(self, cache) -> None:
        super().setup(cache)
        t = self._capacity(cache)
        if t not in self._neg:
            # 32-byte ROW_MAJOR pages; INDEXER_GLUE=on: 64-byte (the score op reads the bias in DRAM-aligned windows)
            al = 32 if self.glue else 16
            nb = -(-(t // self.block) // al) * al
            self._neg[t] = _upload(
                self.mesh, torch.full((1, 1, self.sq, nb), float("-inf")), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )

    def bias_of(self, cands: ttnn.Tensor, kv_len: int, t: int) -> ttnn.Tensor:
        """The chip's candidate bias [1, 1, Sq, kv_len] bf16 ROW_MAJOR (0 on candidate keys, -inf elsewhere)."""
        b, sq = self.block, self.sq
        nb = -(-(kv_len // b) // 16) * 16
        neg = self._neg[t]
        base = ttnn.slice(neg, [0, 0, 0, 0], [1, 1, sq, nb], memory_config=DRAM)
        blk = ttnn.scatter(base, -1, cands, self.src, memory_config=DRAM)  # does not write its input
        if not _same(base, neg):  # K-1 / DS41-52: never free the load-time row
            dealloc(base)
        full = ttnn.repeat_interleave(blk, b, dim=-1, memory_config=DRAM)  # [1, 1, Sq, nb b]
        dealloc(blk)
        if nb * b == kv_len:
            return full
        out = ttnn.slice(full, [0, 0, 0, 0], [1, 1, sq, kv_len], memory_config=DRAM)
        dealloc(full)
        return out

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache, cands: ttnn.Tensor) -> ttnn.Tensor:
        """attn_norm, q_resid as TtIndexer; ``cands`` the candidate source's pool of the chip's Sq query rows
        [1, 1, Sq, 2048] uint32 ROW_MAJOR (read, not consumed) -> top-k [1, 1, W, 512] uint32, gathered over TP."""
        t = self._capacity(cache)
        if self.glue:  # INDEXER_GLUE=on (P.16): the score op applies the block bias itself (fork block epilogue)
            brm = ttnn.scatter(self._neg[t], -1, cands, self.src, memory_config=DRAM)  # [1, 1, Sq, NB]: 0 / -inf
            bias = ttnn.to_layout(brm, ttnn.TILE_LAYOUT, memory_config=DRAM)  # the op reads it by tiles
            dealloc(brm)
            logits, kv_len = self.scores(x, qr, start, cache, block_keys=self.block, block_bias=bias)
            dealloc(bias)
            out = self.topk_of(logits, kv_len)
            dealloc(logits)
            return out
        logits, kv_len = self.scores(x, qr, start, cache)
        lg = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, self.sq, kv_len], memory_config=DRAM)
        bias = self.bias_of(cands, kv_len, t)
        masked = ttnn.add(lg, bias, memory_config=DRAM)  # ROW_MAJOR bf16 [1, 1, Sq, kv_len]
        if not _same(lg, logits):  # K-1 / DS41-52: a full-range slice is a wrapper over the scores' buffer
            dealloc(lg)
        dealloc(bias)
        dealloc(logits)
        out = self.topk_of(masked, kv_len)
        dealloc(masked)
        return out


def indexer_weights(ck, layer: int) -> dict[str, torch.Tensor]:
    p = f"layers.{layer}.attn.indexer."
    return {
        "wq_b": ck.weight(p + "wq_b.weight", torch.float32),
        "weights_proj": ck.weight(p + "weights_proj.weight", torch.float32),
    }


def build_indexer(
    mesh, ck, cfg, layer: int, max_seq: int, chunk: int, rope: RopeTable | None = None, full_rows: bool = False
) -> TtIndexer:
    """The indexer of an index-source layer (enc_source: own keys; the candidate source, layer 20: TtCandidateIndexer;
    the re-index layers over layer 20's keys and pool: TtReindexer) with its own RoPE table (component tests); the
    assembled model passes the layer's shared table."""
    from .kv import rope_freqs

    rope = rope or RopeTable(mesh, rope_freqs(cfg, layer), max_seq)
    if cfg.is_cand_source(layer):
        cls = TtCandidateIndexer
    elif cfg.uses_candidates(layer):
        cls = TtReindexer
    else:
        cls = TtIndexer
    return cls(mesh, indexer_weights(ck, layer), cfg, cfg.ratio(layer), rope, chunk, full_rows=full_rows)
