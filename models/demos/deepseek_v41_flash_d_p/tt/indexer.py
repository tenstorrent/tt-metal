# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 indexer of an index-source layer with its own compressed index keys (reference/ds41_ref.py:_indexer,
enc_source layers 2 / 8 / 14: ratio 2) on the SP x TP mesh. Adapted from models/demos/hy4_preview_d_p/tt/indexer.py
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

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int, cache) -> ttnn.Tensor:
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
        local = ttnn.experimental.topk_large_indices(logits, k=self.topk, valid_length=kv_len)
        ttnn.deallocate(logits)
        if self.tp == 1:
            return local
        out = ttnn.all_gather(local, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)  # [1, 1, W, topk]
        ttnn.deallocate(local)
        return out


def indexer_weights(ck, layer: int) -> dict[str, torch.Tensor]:
    p = f"layers.{layer}.attn.indexer."
    return {
        "wq_b": ck.weight(p + "wq_b.weight", torch.float32),
        "weights_proj": ck.weight(p + "weights_proj.weight", torch.float32),
    }


def build_indexer(mesh, ck, cfg, layer: int, max_seq: int, chunk: int, rope: RopeTable | None = None) -> TtIndexer:
    """The indexer of an enc_source layer with its own RoPE table (component tests); the assembled model passes the
    layer's shared table."""
    from .kv import rope_freqs

    rope = rope or RopeTable(mesh, rope_freqs(cfg, layer), max_seq)
    return TtIndexer(mesh, indexer_weights(ck, layer), cfg, cfg.ratio(layer), rope, chunk)
