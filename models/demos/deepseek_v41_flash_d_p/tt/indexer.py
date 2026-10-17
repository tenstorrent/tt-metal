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

Per chip (r, c), inputs attn_norm [1, 1, W, H/TP] fp32 (hidden split) and q_resid [1, 1, W, 1280] fp32 (row split,
replicated over TP), W = S / SP; the chip's queries are rows [c Sq, (c+1) Sq) of its row block, Sq = W / TP, i.e.
positions start + r W + c Sq + [0, Sq) (natural blocks, any 32-aligned start):

    w   attn_norm (rounded to TF32, setting COMPRESSOR_X: the compressor's fused fp32 projection reads it the same
        way) @ the chip's K rows of (weights_proj / 64)^T (bf16, exact: 1/64 = 2^-6) -> tt/common.tp_sum (exact
        fp32, K-30) -> the chip's Sq rows (mesh_partition over axis 1) -> bf16 (the score op's gate dtype)
    q   the chip's Sq rows of q_resid -> tt/quant.fp8_qdq (bf16, exact) @ wq_b^T (replicated, all 32 heads; bf16:
        e4m3 x 2^k is bf16-exact; every head's rope rows de-interleaved at load, tt/rope.deinterleave_perm) at
        HiFi4 + fp32 acc -> [1, 32, Sq, 128] fp32 -> tt/rope.rope_heads (fp32 SFPU, the reference's interleaved
        order back) -> tt/quant.fp4_qdq(32, ue8m0) -> bf16 (exact)
    k   the layer's served index-key cache (tt/compressor.TtServedRows, 2_index_k_r2: SP-sharded, TP-replicated,
        bf16 ROW_MAJOR in token units, [rows, 64] per chip; the compressor wrote this chunk's keys) viewed as
        compressed rows [rows / 2, 128] and tiled: block-cyclic over SP with W / 2 compressed rows per SP rank per
        chunk (position-based: token g on mesh row (g / W) % SP)
    score ttnn.bringup.ring_indexer_score_dsa (fork indexer_score: fp32 DEST + HiFi4, k_chunk 32, K-13 / K-14),
        ringing the keys over axis 0, key_compression_ratio 2 (fork option: causal t < (pos + 1) // 2) and
        natural_query_blocks (fork option: queries start + r W + c Sq + i at any tile-aligned start while the keys
        keep the cache's block-cyclic layout) -> [1, 1, Sq, T] bf16 ROW_MAJOR, columns >= kv_len unwritten
    top-k ttnn.experimental.topk_large_indices(k 512, valid_length = kv_len) -> [1, 1, Sq, 512] uint32 (descending
        score, -inf -> 0xFFFFFFFF tail) -> all_gather over axis 1 -> [1, 1, W, 512], replicated over TP

The key buffer the ring gathers into ([1, 1, T, 128], T = the cache's compressed capacity) is allocated once per
cache capacity (``setup``). No host work in ``__call__``: the per-chunk inputs are integers (start) and device
tensors. Ties: the scores are bf16 (topk_large_indices is bf16-only), so near ties flip where the reference's fp32
scores do not (plan.md "Accuracy notes", K-42).
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, SP_AXIS, TP_AXIS, compute_config, tp_sum
from .quant import fp4_qdq, fp8_qdq, round_tf32
from .rope import RopeTable, deinterleave_perm, rope_heads
from .settings import settings

Q_CHUNK, K_CHUNK = 64, 32  # score work unit: k 32 is the fork's fidelity-honouring head reduction (K-13 / K-14)
Q_BLOCK = 32  # fp4 block of the index query (ue8m0 scale per block)


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
    positions; 0xFFFFFFFF = none), row-split, replicated over TP. Does not consume its inputs."""

    def __init__(self, mesh, w: dict[str, torch.Tensor], cfg, ratio: int, rope: RopeTable, chunk: int):
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

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
        self.ckc = compute_config(mesh)
        self.round_x = settings.get("COMPRESSOR_X") == "round_tf32"
        # wq_b [nh * hd, q_lora] -> W^T [q_lora, nh * hd], every head's rope rows de-interleaved; replicated
        perm = deinterleave_perm(self.hd, self.rd)
        wq = w["wq_b"].float()
        q_lora = wq.shape[1]
        wq = wq.view(self.nh, self.hd, q_lora)[:, perm, :].reshape(self.nh * self.hd, q_lora)
        wq_t = wq.t().contiguous()
        assert torch.equal(wq_t.bfloat16().float(), wq_t), "wq_b must be bf16-exact (e4m3 x 2^k)"
        self.wq_b = _upload(mesh, wq_t.reshape(1, 1, q_lora, -1), ttnn.bfloat16)
        # weights_proj [nh, H] scaled by hd^-0.5 nh^-0.5 (a power of two: exact) -> W^T [H, nh], K rows split over TP
        scale = self.hd**-0.5 * self.nh**-0.5
        wp = w["weights_proj"].float() * scale
        assert torch.equal(wp.bfloat16().float(), wp), "weights_proj x scale must stay bf16-exact"
        hidden = wp.shape[1]
        assert hidden % (32 * self.tp) == 0, (hidden, self.tp)
        dims = (None, 2) if TP_AXIS == 1 else (2, None)
        self.wproj = _upload(mesh, wp.t().reshape(1, 1, hidden, self.nh), ttnn.bfloat16, dims=dims)
        self.ccl = get_tt_ccl(mesh)
        self.topology = ttnn.Topology.Linear  # FABRIC_2D: no wrap assumed on the SP axis (every box)
        self.num_links = 2 if mesh.arch() == ttnn.Arch.BLACKHOLE else 1
        self.score_cfg = ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=min(Q_CHUNK, self.sq), k_chunk_size=K_CHUNK, head_group_size=0
        )
        self.score_ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=bool(settings.get("FP32_DEST_ACC")),
            packer_l1_acc=False,
        )
        self._kbuf = {}  # compressed capacity T -> the ring's persistent gathered-key buffer [1, 1, T, hd]

    # ------------------------------------------------------------------ load time
    def setup(self, cache) -> None:
        """The ring's gathered-key buffer for this cache's capacity (allocated once per capacity)."""
        t = self._capacity(cache)
        if t not in self._kbuf:
            self._kbuf[t] = ttnn.from_torch(
                torch.zeros(1, 1, t, self.hd),
                dtype=ttnn.bfloat16,
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
    def _keys(self, cache) -> ttnn.Tensor:
        """The served index-key cache [1, 1, rows, hd / r] bf16 ROW_MAJOR (token units, ND-sharded) -> compressed
        rows [1, 1, rows / r, hd] bf16 TILE, interleaved: the ring's k_local (block-cyclic layout unchanged)."""
        rm = ttnn.to_memory_config(cache.tensor, DRAM)
        rows = rm.shape[-2]
        view = ttnn.reshape(rm, [1, 1, rows // self.ratio, self.hd])
        kt = ttnn.to_layout(view, ttnn.TILE_LAYOUT, memory_config=DRAM)
        if rm is not cache.tensor:  # an interleaved cache would come back as itself (K-1): never free the cache
            ttnn.deallocate(rm)
        return kt

    def _sub(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """This chip's Sq query rows of a row-split, TP-replicated [1, 1, W, n] tensor."""
        if self.tp == 1:
            return ttnn.clone(t, memory_config=DRAM)
        return ttnn.mesh_partition(t, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)

    def gates(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """attn_norm [1, 1, W, H/TP] fp32 -> the chip's per-head gates [1, 1, Sq, nh] bf16."""
        xa = round_tf32(x) if self.round_x else x
        part = ttnn.linear(xa, self.wproj, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        if xa is not x:
            ttnn.deallocate(xa)
        full = tp_sum(self.mesh, part)
        ttnn.deallocate(part)
        sub = self._sub(full)
        ttnn.deallocate(full)
        out = ttnn.typecast(sub, ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(sub)
        return out

    def query(self, qr: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """q_resid [1, 1, W, q_lora] fp32 -> the chip's index queries [1, nh, Sq, hd] bf16 (fp4 values)."""
        sub = self._sub(qr)
        qq = fp8_qdq(sub, dtype=ttnn.bfloat16)
        ttnn.deallocate(sub)
        q = ttnn.linear(qq, self.wq_b, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(qq)
        q4 = ttnn.reshape(q, [1, self.sq, self.nh, self.hd])
        qh = ttnn.permute(q4, (0, 2, 1, 3), memory_config=DRAM)  # [1, nh, Sq, hd]
        ttnn.deallocate(q)
        cos_r, sin_r = self.rope.rows(start, self.chunk)  # the row's W positions
        cos, sin = self._sub(cos_r), self._sub(sin_r)
        for t in (cos_r, sin_r):
            ttnn.deallocate(t)
        qr_ = rope_heads(qh, cos, sin, interleave=True)
        for t in (qh, cos, sin):
            ttnn.deallocate(t)
        out = fp4_qdq(qr_, Q_BLOCK, dtype=ttnn.bfloat16)
        ttnn.deallocate(qr_)
        return out

    def scores(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> tuple[ttnn.Tensor, int]:
        """The chip's index scores [1, 1, Sq, T] bf16 ROW_MAJOR (future columns -inf, columns >= kv_len unwritten)
        and kv_len (the written width, a multiple of 32)."""
        assert start % 32 == 0, start
        t = self._capacity(cache)
        kbuf = self._kbuf[t]  # setup(cache) at load
        end = start + self.chunk  # the padded window's end; later keys are causally masked for every real query
        kv_len = min(t, -(-(end // self.ratio) // 32) * 32)
        assert kv_len >= self.topk and start < kv_len * self.ratio, (start, kv_len, self.topk)
        w = self.gates(x)
        q = self.query(qr, start)
        k_local = self._keys(cache)
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
        )  # [1, 1, Sq, T] bf16 ROW_MAJOR; future columns -inf, columns >= kv_len unwritten
        for u in (q, w, k_local):
            ttnn.deallocate(u)
        return logits, kv_len

    def topk_of(self, logits: ttnn.Tensor, kv_len: int) -> ttnn.Tensor:
        """The 512 best keys per query of the chip's scores -> [1, 1, W, topk] uint32, gathered over TP."""
        local = ttnn.experimental.topk_large_indices(logits, k=self.topk, valid_length=kv_len)
        if self.tp == 1:
            return local
        out = ttnn.all_gather(local, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)  # [1, 1, W, topk]
        ttnn.deallocate(local)
        return out

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> ttnn.Tensor:
        logits, kv_len = self.scores(x, qr, start, cache)
        out = self.topk_of(logits, kv_len)
        ttnn.deallocate(logits)
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
    within a row is free (index sets; ties: bf16 scores, K-42)."""

    def __init__(self, mesh, w, cfg, ratio: int, rope: RopeTable, chunk: int):
        super().__init__(mesh, w, cfg, ratio, rope, chunk)
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

    def setup(self, cache) -> None:
        super().setup(cache)
        t = self._capacity(cache)
        if t not in self._cols:
            nb = -(-(t // self.block) // 32) * 32
            self._cols[t] = _upload(self.mesh, torch.arange(nb, dtype=torch.float32).reshape(1, 1, 1, nb), ttnn.float32)

    def candidates_of(self, logits: ttnn.Tensor, kv_len: int, start: int, t: int) -> ttnn.Tensor:
        """The chip's candidate blocks [1, 1, Sq, ctopk] uint32 from its scores (does not consume them)."""
        b = self.block
        nb = -(-(kv_len // b) // 32) * 32  # block columns, tile-aligned
        sq = self.sq
        lg = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, sq, kv_len], memory_config=DRAM)  # logits itself if kv_len == T
        if nb * b > kv_len:
            padded = ttnn.pad(lg, [(0, 0), (0, 0), (0, 0), (0, nb * b - kv_len)], value=float("-inf"))
            if not _same(lg, logits):  # K-1: a full-range slice returns its input's buffer
                ttnn.deallocate(lg)
            lg = padded
        acc = None
        for i in range(b):  # block max: column j of slice i = score j b + i
            part = ttnn.slice(lg, [0, 0, 0, i], [1, 1, sq, nb * b], [1, 1, 1, b], memory_config=DRAM)
            if acc is None:
                acc = part
                continue
            nxt = ttnn.maximum(acc, part, memory_config=DRAM)
            ttnn.deallocate(acc)
            ttnn.deallocate(part)
            acc = nxt
        if not _same(lg, logits):
            ttnn.deallocate(lg)
        blk = ttnn.to_layout(acc, ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, 1, Sq, nb] bf16
        if not _same(blk, acc):  # K-1
            ttnn.deallocate(acc)
        # pin: block (start + row offset) // b of every query row
        pos = ttnn.add(self.row_pos, float(start), memory_config=DRAM)
        pin = ttnn.floor(ttnn.multiply(pos, 1.0 / b, memory_config=DRAM), memory_config=DRAM)  # b a power of two
        ttnn.deallocate(pos)
        cols = ttnn.slice(self._cols[t], [0, 0, 0, 0], [1, 1, 1, nb], memory_config=DRAM)
        hit = ttnn.eq(cols, pin, dtype=ttnn.float32, memory_config=DRAM)  # [1, 1, Sq, nb] 1 at the pinned block
        ttnn.deallocate(pin)
        if not _same(cols, self._cols[t]):  # K-1: never free the load-time iota
            ttnn.deallocate(cols)
        boost = ttnn.multiply(hit, BIG, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(hit)
        pinned = ttnn.add(blk, boost, dtype=ttnn.bfloat16, memory_config=DRAM)
        for u in (blk, boost):
            ttnn.deallocate(u)
        rm = ttnn.to_layout(pinned, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if not _same(rm, pinned):
            ttnn.deallocate(pinned)
        if nb < self.ctopk:  # fewer blocks than k: the rest are none (-inf -> 0xFFFFFFFF)
            padded = ttnn.pad(rm, [(0, 0), (0, 0), (0, 0), (0, self.ctopk - nb)], value=float("-inf"))
            ttnn.deallocate(rm)
            rm = padded
        out = ttnn.experimental.topk_large_indices(rm, k=self.ctopk)
        ttnn.deallocate(rm)
        return out

    def forward(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        logits, kv_len = self.scores(x, qr, start, cache)
        cands = self.candidates_of(logits, kv_len, start, self._capacity(cache))
        out = self.topk_of(logits, kv_len)
        ttnn.deallocate(logits)
        return out, cands

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> ttnn.Tensor:
        out, cands = self.forward(x, qr, start, cache)
        ttnn.deallocate(cands)
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
    reference would count a masked key (its ``idx < lens`` keeps them) except below 512 visible keys, where both pad."""

    def __init__(self, mesh, w, cfg, ratio: int, rope: RopeTable, chunk: int):
        super().__init__(mesh, w, cfg, ratio, rope, chunk)
        assert ratio == 1, "the re-index layers score the candidate source's key positions (ratio 1)"
        self.block, self.ctopk = cfg.cand_block, cfg.cand_topk_blocks
        assert 32 % self.block == 0, self.block  # kv_len (a multiple of 32) holds whole blocks
        self.src = _upload(mesh, torch.zeros(1, 1, self.sq, self.ctopk), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._neg = {}  # capacity T -> the -inf block row [1, 1, Sq, ceil16(T / block)] bf16 ROW_MAJOR

    def setup(self, cache) -> None:
        super().setup(cache)
        t = self._capacity(cache)
        if t not in self._neg:
            nb = -(-(t // self.block) // 16) * 16  # 32-byte ROW_MAJOR pages
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
            ttnn.deallocate(base)
        full = ttnn.repeat_interleave(blk, b, dim=-1, memory_config=DRAM)  # [1, 1, Sq, nb b]
        ttnn.deallocate(blk)
        if nb * b == kv_len:
            return full
        out = ttnn.slice(full, [0, 0, 0, 0], [1, 1, sq, kv_len], memory_config=DRAM)
        ttnn.deallocate(full)
        return out

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache, cands: ttnn.Tensor) -> ttnn.Tensor:
        """attn_norm, q_resid as TtIndexer; ``cands`` the candidate source's pool of the chip's Sq query rows
        [1, 1, Sq, 2048] uint32 ROW_MAJOR (read, not consumed) -> top-k [1, 1, W, 512] uint32, gathered over TP."""
        t = self._capacity(cache)
        logits, kv_len = self.scores(x, qr, start, cache)
        lg = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, self.sq, kv_len], memory_config=DRAM)
        bias = self.bias_of(cands, kv_len, t)
        masked = ttnn.add(lg, bias, memory_config=DRAM)  # ROW_MAJOR bf16 [1, 1, Sq, kv_len]
        if not _same(lg, logits):  # K-1 / DS41-52: a full-range slice is a wrapper over the scores' buffer
            ttnn.deallocate(lg)
        ttnn.deallocate(bias)
        ttnn.deallocate(logits)
        out = self.topk_of(masked, kv_len)
        ttnn.deallocate(masked)
        return out


def indexer_weights(ck, layer: int) -> dict[str, torch.Tensor]:
    p = f"layers.{layer}.attn.indexer."
    return {
        "wq_b": ck.weight(p + "wq_b.weight", torch.float32),
        "weights_proj": ck.weight(p + "weights_proj.weight", torch.float32),
    }


def build_indexer(mesh, ck, cfg, layer: int, max_seq: int, chunk: int, rope: RopeTable | None = None) -> TtIndexer:
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
    return cls(mesh, indexer_weights(ck, layer), cfg, cfg.ratio(layer), rope, chunk)
