# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 DSA lightning indexer on the 2x2 mesh (SP=2 over axis 0 x TP=2 over axis 1, plan.md).

    q     = RoPE(wq_b(q_resid))            [S, 32, 128], RoPE (interleaved, theta 1e7) on dims 64..127
    k     = RoPE(LayerNorm(wk(attn_norm)))  [S, 128], eps 1e-5, with bias, RoPE on dims 64..127
    w     = weights_proj(attn_norm) * 32^-0.5 * 128^-0.5
    score = sum_h w[s, h] relu(q[s, h] . k[t]) for t <= s;  topk 2048 (unsorted, 0xFFFFFFFF sentinel tail)

Adapted from models/demos/deepseek_v3_d_p/tt/mla/indexer.py:TtIndexer (score / write_k / select_local /
finalize_distribution); same ops, same block-cyclic key cache striped over all 4 chips (KV dedup), same fused ring
score over axis 0. Differences (components.yaml, plan.md):

- RoPE on the LAST 64 of the 128 dims: ``rotary_embedding_indexed(rotary_dim=64, rotary_offset=64)`` ropes channels
  64..127 in place and copies 0..63. No host permutation of wq_b / wk / k_norm is needed, and the index-key cache is
  in the reference's (checkpoint) order, so state read-back needs no un-permute.
- k_norm LayerNorm eps is the model's (1e-5; TtIndexer hard-codes 1e-6).
- bf16 index-key cache and bf16 q (TtIndexer: bfp8 via its Hadamard matmul). No Hadamard basis change.
- bf16 wq_b / wk (TtIndexer: bfp8), fp32 weights_proj with the 1/64 scale folded (a power of two: exact); HiFi4 +
  fp32 dest for every ttnn.linear and the RoPE.
- Score: ``ttnn.bringup.ring_indexer_score_dsa`` (the indexer_score fork) at HiFi4 + fp32 DEST, work unit q 64 x k 32.
  The source op requires bf16 DEST, whose MAC over the 32 heads truncates (every logit ~1.7% low), and at k > 32 its
  blocked gate multiply runs at LoFi-like precision whatever the fidelity: logits rel error 0.024 and top-k overlap
  0.985 on the layer-0 golden (gate 0.99). Fork + fp32 DEST + k 32 (the per-column path, fidelity honoured): rel
  0.0036, overlap 0.99708 (fp32 scores rounded to bf16: 0.99705). ``score_impl="native"`` keeps the source op
  (bf16 DEST, q 64 x k 32: overlap 0.9926).
- TP reduces with ttnn.all_reduce over axis 1 on fp32 partials (TtQa's choice) instead of reduce_scatter +
  high_bw_all_gather with owned buffers.

Per chip (r, c), inputs attn_norm [1, 1, S/2, 3072] (column split, bf16) and q_resid [1, 1, S/2, 2048] (row split,
replicated over axis 1, bf16). Output [1, 1, S/2, 2048] uint32 ROW_MAJOR key positions of this chip's S/2 query rows,
replicated over axis 1 (like q_resid). Geometry-dependent constants (RoPE tables in block-cyclic order, the key cache,
the ring and gather scratch) are built once per (chunk, max_seq) by ``setup``; ``__call__`` does no host work.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

TILE = 32
SENTINEL = 0xFFFFFFFF
# Ring score work units (rows x key columns). TtIndexer uses q 64 x k 320 at 32 heads with bfp8 q / k. With bf16 q / k
# and bf16 DEST, k 320 overflows L1 (1672192 B of CBs > 1572864 B); fp32 DEST doubles the qk / accumulator CBs (q 64 x
# k 256: 2049024 B). k 32 is the kernel's per-column head reduction, the only path whose gate multiply honours the
# fidelity (k > 32: blocked custom multiply, LoFi-like); native at k 256 scored overlap 0.9853, at k 32 0.9926.
K_CHUNK_BF16_DEST = 32
Q_CHUNK, K_CHUNK = 64, 32


def rope_tables(positions: torch.Tensor, theta: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved (Meta pair) cos / sin [N, dim]: entries 2i, 2i+1 hold cos / sin of pos * theta^(-2i/dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    freqs = (positions.float()[:, None] * inv_freq[None, :]).repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


class _Geometry:
    """Per (chunk, max_seq) device constants: block-cyclic RoPE tables, the bf16 index-key cache, scratch."""

    def __init__(self, idx: "TtHy4Indexer", chunk: int, max_seq: int):
        mesh, sp, tp = idx.mesh, idx.sp, idx.tp
        assert chunk % (TILE * sp * tp) == 0, f"chunk {chunk} must be a multiple of {TILE * sp * tp}"
        assert max_seq % chunk == 0, f"max_seq {max_seq} must be a multiple of the chunk {chunk}"
        assert chunk >= idx.topk, f"chunk {chunk} < topk {idx.topk}: topk_large_indices needs valid_length >= k"
        self.chunk, self.max_seq = chunk, max_seq
        self.chunk_local = chunk // sp  # query / key rows per SP rank
        cos, sin = rope_tables(torch.arange(max_seq), idx.rope_theta, idx.rope_dim)
        shard_sp = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))

        def table(t):
            t = block_cyclic_reorder(t.reshape(1, 1, max_seq, idx.rope_dim), self.chunk_local, sp, seq_dim=2)
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard_sp,
            )

        self.cos, self.sin = table(cos), table(sin)
        self.trans = ttnn.from_torch(
            get_rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # Index-key cache [1, 1, max_seq / (sp*tp), 128] per chip, block-cyclic over the 4 chips (chip L = r*tp + c
        # holds positions j*chunk + L*chunk/4 + [0, chunk/4) of every slab j), zero-initialised on the device.
        self.cache = init_kvpe_cache(
            idx.head_dim,
            mesh,
            max_seq,
            tuple(mesh.shape),
            idx.sp_axis,
            1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tp_axis=idx.tp_axis,
        )
        self._topology = self.cache.tensor_topology()
        # Scratch: this SP row's full slab rebuilt by a TP-inner gather, the ring's full-T key buffer, and the
        # TP gather of the top-k rows. Allocated once; the ops write them in place.
        self.k_local = ttnn.empty(
            [1, 1, max_seq // sp, idx.head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_full = idx.ccl.get_indexer_ring_k_buffer(local_k=self.k_local, sp_axis=idx.sp_axis)
        self.idx_out = ttnn.empty(
            [1, 1, self.chunk_local, idx.topk],
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sq = self.chunk_local // tp
        self.score_cfg = idx.score_cfg_cls(
            q_chunk_size=min(idx.q_chunk, sq), k_chunk_size=min(idx.k_chunk, chunk), head_group_size=0
        )

    # ---- harness boundary (state load / read-back); never called from __call__
    def load(self, mesh, prefix: torch.Tensor | None, sp: int, tp: int, head_dim: int) -> None:
        """Write natural-order index keys [n, 128] (n <= max_seq, zeros past n) into the block-cyclic cache."""
        nat = torch.zeros(self.max_seq, head_dim, dtype=torch.bfloat16)
        if prefix is not None and prefix.shape[0]:
            nat[: prefix.shape[0]] = prefix.to(torch.bfloat16)
        p = blockcyclic_positions(sp * tp, self.chunk, self.max_seq)  # shard row -> natural position
        bc = nat[p]  # [max_seq, D], stripe L = rows [L*local, (L+1)*local)
        local = self.max_seq // (sp * tp)
        host = bc.reshape(sp, tp, local, head_dim).permute(1, 0, 2, 3).reshape(1, tp, self.max_seq // tp, head_dim)
        ht = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 1)),
        )
        ttnn.copy_host_to_device_tensor(ht, self.cache)
        # copy_host_to_device_tensor stamps the host mapper's [Shard(2), Shard(1)]; restore the cache's own
        # sp*tp sequence distribution (as test_prefill_transformer_chunked._preload_kvpe_prefix_from_trace does).
        self.cache.update_tensor_topology(self._topology)

    def read(self, mesh, length: int, sp: int, tp: int) -> torch.Tensor:
        """The cache in natural order [length, 128] fp32."""
        cache_sr = ttnn.to_torch(
            self.cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 1), mesh_shape=tuple(mesh.shape))
        ).float()  # [1, tp, max_seq / tp, D]
        local = cache_sr.shape[2] // sp
        flat = torch.cat([cache_sr[0, t, s * local : (s + 1) * local] for s in range(sp) for t in range(tp)], dim=0)
        p = blockcyclic_positions(sp * tp, self.chunk, self.max_seq)
        nat = torch.empty_like(flat)
        nat[p] = flat
        return nat[:length]


class TtHy4Indexer:
    """One layer's indexer. ``setup(chunk, max_seq)`` once per geometry, then ``__call__(attn_norm, q_resid, start)``
    per chunk (start a multiple of the chunk); returns the top-k key positions [1, 1, S/2, topk] uint32 ROW_MAJOR
    (sentinel 0xFFFFFFFF for rows with fewer than topk causal keys), replicated over the TP axis."""

    def __init__(
        self,
        mesh,
        wq_b: torch.Tensor,  # [n_heads * head_dim, q_lora_rank]
        wk: torch.Tensor,  # [head_dim, hidden]
        k_norm_w: torch.Tensor,  # [head_dim]
        k_norm_b: torch.Tensor,  # [head_dim]
        weights_proj: torch.Tensor,  # [n_heads, hidden]
        *,
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        rope_theta: float,
        eps: float,
        topk: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        score_impl: str = "bringup",
    ):
        assert rope_dim % TILE == 0 and head_dim - rope_dim >= 0 and (head_dim - rope_dim) % TILE == 0
        self.mesh, self.sp_axis, self.tp_axis = mesh, sp_axis, tp_axis
        self.sp, self.tp = mesh.shape[sp_axis], mesh.shape[tp_axis]
        assert self.tp > 1, "the deduped index-key cache needs a TP axis"
        self.n_heads, self.head_dim, self.rope_dim, self.rope_theta = n_heads, head_dim, rope_dim, float(rope_theta)
        self.eps, self.topk = float(eps), topk
        hidden = wk.shape[1]
        assert (
            wq_b.shape[0] == n_heads * head_dim and wk.shape[0] == head_dim and weights_proj.shape == (n_heads, hidden)
        )
        self.ccl = get_tt_ccl(mesh)
        self.topology = ttnn.Topology.Linear  # FABRIC_2D: no wrap on either axis of the 2x2 mesh
        self.num_links = 2 if mesh.arch() == ttnn.Arch.BLACKHOLE else 1  # as ttMLA.ccl_num_links
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Score op: "bringup" = ttnn.bringup.ring_indexer_score_dsa (indexer_score fork) with fp32 DEST, so q.k, the
        # gate MAC over the 32 heads and the accumulator run in fp32; "native" = ttnn.experimental's op, which
        # requires bf16 DEST (see the module docstring: overlap 0.9853 at k 256, 0.9926 at k 32).
        assert score_impl in ("bringup", "native"), score_impl
        self.score_op = (
            ttnn.bringup.ring_indexer_score_dsa if score_impl == "bringup" else ttnn.experimental.ring_indexer_score_dsa
        )
        # The fork binds its own IndexerScoreProgramConfig type; each op takes only its own.
        self.score_cfg_cls = (
            ttnn.bringup.IndexerScoreProgramConfig if score_impl == "bringup" else ttnn.IndexerScoreProgramConfig
        )
        self.score_ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=score_impl == "bringup",
            packer_l1_acc=False,
        )
        dram = ttnn.DRAM_MEMORY_CONFIG
        ksplit = (None, 2) if tp_axis == 1 else (2, None)  # W^T [hidden, out], K rows split over the TP axis

        def put(t, dtype, dims=None, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh)
                if dims is None
                else ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)
            )
            return ttnn.from_torch(
                t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=mapper
            )

        q_lora = wq_b.shape[1]
        self.wq_b = put(wq_b.t().reshape(1, 1, q_lora, n_heads * head_dim), ttnn.bfloat16)  # replicated, all heads
        self.wk = put(wk.t().reshape(1, 1, hidden, head_dim), ttnn.bfloat16, ksplit)
        scale = n_heads**-0.5 * head_dim**-0.5
        self.wproj = put((weights_proj.float() * scale).t().reshape(1, 1, hidden, n_heads), ttnn.float32, ksplit)
        self.k_norm_w = put(
            k_norm_w.float().reshape(1, 1, head_dim // TILE, TILE), ttnn.bfloat16, None, ttnn.ROW_MAJOR_LAYOUT
        )
        self.k_norm_b = put(
            k_norm_b.float().reshape(1, 1, head_dim // TILE, TILE), ttnn.bfloat16, None, ttnn.ROW_MAJOR_LAYOUT
        )
        self.q_chunk, self.k_chunk = (Q_CHUNK, K_CHUNK) if score_impl == "bringup" else (64, K_CHUNK_BF16_DEST)
        self._geoms: dict = {}
        self.geom: _Geometry | None = None
        self.slot = None  # serving binding (bind_cache); None = the geometry's own single-sequence cache

    # ---- load time
    def bind_cache(self, cache, slot: int, row: int, rows: int) -> None:
        """Serving option: write / read this layer's index keys in an external multi-slot cache (the prefill
        engine's, tt/runners/kv_contract.py) instead of the geometry's own. ``cache`` is laid out like the geometry's
        (init_kvpe_cache, tp_axis=1, same max_seq and chunk) with batch = slot * rows + row. ``unbind_cache``
        (the default) restores the geometry cache at batch 0."""
        assert tuple(cache.shape)[1:] == tuple(self.geom.cache.shape)[1:], (cache.shape, self.geom.cache.shape)
        assert 0 <= row < rows and (slot + 1) * rows <= cache.shape[0], (slot, row, rows, cache.shape)
        self.slot = (cache, int(slot), int(row), int(rows))

    def unbind_cache(self) -> None:
        self.slot = None

    def _cache(self):
        """(cache, slot_idx, layer_idx, num_layers) the chunk writes and gathers."""
        return self.slot if self.slot is not None else (self.geom.cache, 0, 0, 1)

    def setup(self, chunk: int, max_seq: int) -> _Geometry:
        key = (chunk, max_seq)
        if key not in self._geoms:
            self._geoms[key] = _Geometry(self, chunk, max_seq)
        self.geom = self._geoms[key]
        return self.geom

    def load_state(self, prefix: torch.Tensor | None) -> None:
        """Harness boundary: natural-order index keys [n, 128] into the current geometry's cache (zeros past n)."""
        self.geom.load(self.mesh, prefix, self.sp, self.tp, self.head_dim)

    def read_state(self, length: int) -> torch.Tensor:
        """Harness boundary: the index-key cache [length, 128] in natural (checkpoint dim) order."""
        return self.geom.read(self.mesh, length, self.sp, self.tp)

    # ---- forward (device only)
    def _rope(self, x, start, subshard=None):
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            self.geom.cos,
            self.geom.sin,
            self.geom.trans,
            start,
            self.sp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ckc,
            seq_subshard_axis=subshard,
            rotary_dim=self.rope_dim,
            rotary_offset=self.head_dim - self.rope_dim,
        )

    def write_k(self, x, start: int) -> None:
        """k = RoPE(LayerNorm(wk(x))) for this chunk -> the index-key cache (this chip's 1/(sp*tp) window)."""
        dram, g = ttnn.DRAM_MEMORY_CONFIG, self.geom
        part = ttnn.linear(x, self.wk, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        k = ttnn.all_reduce(part, cluster_axis=self.tp_axis, memory_config=dram)  # [1, 1, S/2, 128] fp32
        ttnn.deallocate(part)
        kn = ttnn.layer_norm(
            k,
            weight=self.k_norm_w,
            bias=self.k_norm_b,
            epsilon=self.eps,
            memory_config=dram,
            compute_kernel_config=self.ckc,
        )
        ttnn.deallocate(k)
        if kn.dtype != ttnn.bfloat16:
            kb = ttnn.typecast(kn, ttnn.bfloat16, memory_config=dram)
            ttnn.deallocate(kn)
            kn = kb
        kr = self._rope(kn, start)
        ttnn.deallocate(kn)
        cache, slot, row, rows = self._cache()
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            kr,
            slot_idx=slot,
            layer_idx=row,
            num_layers=rows,
            kv_actual_global=start,
            cluster_axis=self.sp_axis,
            tp_axis=self.tp_axis,
        )
        ttnn.deallocate(kr)

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int) -> ttnn.Tensor:
        dram, g = ttnn.DRAM_MEMORY_CONFIG, self.geom
        assert g is not None, "call setup(chunk, max_seq) first"
        assert start % g.chunk == 0, f"start {start} must be chunk ({g.chunk}) aligned"
        end = start + g.chunk
        self.write_k(x, start)

        # q: this TP rank's quarter of the rows, all 32 heads (wq_b replicated) -> [1, 32, S/4, 128] bf16, RoPE'd.
        qr_local = ttnn.mesh_partition(qr, dim=2, cluster_axis=self.tp_axis)
        q = ttnn.linear(qr_local, self.wq_b, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(qr_local)
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.n_heads, num_kv_heads=0, transpose_k_heads=False, memory_config=dram
        )
        ttnn.deallocate(q)
        qh_r = self._rope(qh, start, subshard=self.tp_axis)
        ttnn.deallocate(qh)

        # Per-head gates: K-split fp32 partials -> all_reduce over TP -> this TP rank's rows -> bf16 [1, 1, S/4, 32].
        wp = ttnn.linear(x, self.wproj, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        wa = ttnn.all_reduce(wp, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(wp)
        wl = ttnn.mesh_partition(wa, dim=2, cluster_axis=self.tp_axis)
        ttnn.deallocate(wa)
        w = ttnn.typecast(wl, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(wl)

        # This SP row's key slab (the two TP stripes, block-cyclic order kept), then the fused ring score over SP.
        cache, slot, row, rows = self._cache()
        ttnn.experimental.high_bw_all_gather(
            cache,
            dim=2,
            output_tensor=g.k_local,
            num_links=self.num_links,
            cluster_axis=self.tp_axis,
            input_batch_index=slot * rows + row,
        )
        logits = self.score_op(
            qh_r,
            g.k_full,
            w,
            g.k_local,
            self.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            cluster_axis=self.sp_axis,
            topology=self.topology,
            num_links=self.num_links,
            chunk_start_idx=start,
            program_config=g.score_cfg,
            compute_kernel_config=self.score_ckc,
            kv_len=end,
            seq_subshard_axis=self.tp_axis,
            block_cyclic_sp_axis=self.sp_axis,
            block_cyclic_chunk_local=g.chunk_local,
            block_cyclic_cache_tp_sharded=True,
        )  # [1, 1, S/4, max_seq] bf16 ROW_MAJOR, future columns -inf, columns >= end stale
        ttnn.deallocate(qh_r)
        ttnn.deallocate(w)
        local = ttnn.experimental.topk_large_indices(logits, k=self.topk, valid_length=end)  # [1, 1, S/4, k] uint32
        ttnn.deallocate(logits)
        out = ttnn.experimental.high_bw_all_gather(
            local, dim=2, output_tensor=g.idx_out, num_links=self.num_links, cluster_axis=self.tp_axis
        )  # [1, 1, S/2, k], replicated over TP
        ttnn.deallocate(local)
        return out
