# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 gated sparse MLA (DSA attention) on the 2x2 mesh (SP=2 over axis 0 x TP=2 over axis 1, plan.md).

    q        = q_b_proj(q_resid) [S, 64, 256] -> q_nope 192 | q_rope 64 (RoPE)
    kv       = kv_a_proj_with_mqa(attn_norm) [S, 576] -> kv_a_layernorm(latent 512, eps 1e-6) | RoPE(k_rope 64)
    q_lat    = q_nope @ W_uk (192 -> 512 per head); q_abs = [q_lat | RoPE(q_rope)] (576)
    o_lat    = softmax over topk keys + sink logit (mass dropped) of q_abs . kv_j / 16, times latent_j (512)
    o        = W_uv o_lat (512 -> 256 per head) * sigmoid(linear_gate(attn_norm))
    attn_out = o_proj(o)

RoPE is interleaved (GPT-J pairs, theta 1e7) in checkpoint order, so ``rotary_embedding_indexed`` (Meta pair order
with the rotation trans-mat) needs no host permutation.

Adapted from models/demos/deepseek_v3_d_p/tt/mla/mla.py:ttMLA (_q_stem, _kv_stem, _update_kv_cache,
_gather_kvpe_prefix, _sparse_mla, _o_proj_epilogue with the Kimi-K3 output gate); same ops, same cache layout
(BF16_RM MLA latent cache striped block-cyclic over all 4 chips, KV dedup). Differences:

- The sink: ``sparse_sdpa(attention_sink=sink * 16, scale=1/16)``. HF adds the raw sink as one more logit; the op
  multiplies the sink by the scale (sparse_sdpa.hpp), and 1/16 is a power of two, so both are exact in bf16.
- bf16 weights (as stored), not ttMLA's bfp8; every matmul, the norm, the RoPE and sparse_sdpa at HiFi4 + fp32 dest.
- kv_a_proj: K-split fp32 partials + ttnn.all_reduce over axis 1 (TtQa's choice) instead of the high_bw_all_gather
  + fast_reduce_nc_split pair with owned buffers; kv_a_layernorm through ``ttnn.bringup.rms_norm`` (the native
  op scales rows ~0.1% low on fp32 input, known issues), with the Hy4 eps 1e-6.
- The output gate: ttnn.all_gather of attn_norm over axis 1 -> linear_gate (head-split columns) -> sigmoid, on the
  concatenated heads before o_proj (Kimi-K3's placement).
- o_proj: row-parallel fp32 partials -> ttnn.reduce_scatter over axis 1 -> attn_out [1, 1, S/2, 3072] fp32 (the
  residual's column split).

Per chip (r, c): inputs attn_norm [1, 1, S/2, 3072] (column split, bf16), q_resid [1, 1, S/2, 2048] (row split,
replicated over axis 1, bf16), topk [1, 1, S/2, 2048] uint32 ROW_MAJOR (natural key positions, 0xFFFFFFFF tail,
replicated over axis 1: the indexer's output). Heads 32c .. 32c + 31 on chip column c. Geometry-dependent constants
(block-cyclic RoPE tables, the latent cache, the gather scratch) are built once per (chunk, max_seq) by ``setup``;
``__call__`` does no host work.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

TILE = 32
SPARSE_K_CHUNK = 128  # sparse_sdpa key chunk: a multiple of 32 dividing topk (2048), as ttMLA


def rope_tables(positions: torch.Tensor, theta: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved (Meta pair) cos / sin [N, dim]: entries 2i, 2i+1 hold cos / sin of pos * theta^(-2i/dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    freqs = (positions.float()[:, None] * inv_freq[None, :]).repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


class _Geometry:
    """Per (chunk, max_seq) device constants: block-cyclic RoPE tables, the bf16 latent cache, the gather scratch."""

    def __init__(self, att: "TtHy4Attention", chunk: int, max_seq: int):
        mesh, sp, tp = att.mesh, att.sp, att.tp
        assert chunk % (TILE * sp * tp) == 0, f"chunk {chunk} must be a multiple of {TILE * sp * tp}"
        assert max_seq % chunk == 0, f"max_seq {max_seq} must be a multiple of the chunk {chunk}"
        self.chunk, self.max_seq = chunk, max_seq
        self.chunk_local = chunk // sp  # query / key rows per SP rank
        cos, sin = rope_tables(torch.arange(max_seq), att.rope_theta, att.rope_dim)
        shard_sp = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))

        def table(t):
            t = block_cyclic_reorder(t.reshape(1, 1, max_seq, att.rope_dim), self.chunk_local, sp, seq_dim=2)
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
        # MLA latent cache [1, 1, max_seq / (sp*tp), 576] bf16 ROW_MAJOR per chip (MlaKvCacheFormat.BF16_RM),
        # block-cyclic over the 4 chips (chip L = r*tp + c holds positions j*chunk + L*chunk/4 + [0, chunk/4) of every
        # slab j), zero-initialised on the device.
        self.cache = init_kvpe_cache(
            att.kv_width,
            mesh,
            max_seq,
            tuple(mesh.shape),
            att.sp_axis,
            1,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            tp_axis=att.tp_axis,
        )
        self._topology = self.cache.tensor_topology()
        # Replicated scratch the full-mesh gather writes the populated prefix into (block-cyclic order kept;
        # sparse_sdpa remaps natural positions in-kernel). Allocated once.
        self.kv_all = ttnn.from_torch(
            torch.zeros(1, 1, max_seq, att.kv_width),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    # ---- harness boundary (state load / read-back); never called from __call__
    def load(self, mesh, prefix: torch.Tensor | None, sp: int, tp: int, width: int) -> None:
        """Write natural-order latent rows [n, 576] (n <= max_seq, zeros past n) into the block-cyclic cache."""
        nat = torch.zeros(self.max_seq, width, dtype=torch.bfloat16)
        if prefix is not None and prefix.shape[0]:
            nat[: prefix.shape[0]] = prefix.to(torch.bfloat16)
        p = blockcyclic_positions(sp * tp, self.chunk, self.max_seq)  # shard row -> natural position
        bc = nat[p]  # [max_seq, W], stripe L = rows [L*local, (L+1)*local)
        local = self.max_seq // (sp * tp)
        host = bc.reshape(sp, tp, local, width).permute(1, 0, 2, 3).reshape(1, tp, self.max_seq // tp, width)
        ht = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 1)),
        )
        ttnn.copy_host_to_device_tensor(ht, self.cache)
        # copy_host_to_device_tensor stamps the host mapper's [Shard(2), Shard(1)]; restore the cache's own
        # sp*tp sequence distribution (the full-mesh gather validates it).
        self.cache.update_tensor_topology(self._topology)

    def read(self, mesh, length: int, sp: int, tp: int) -> torch.Tensor:
        """The cache in natural order [length, 576] fp32."""
        cache_sr = ttnn.to_torch(
            self.cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 1), mesh_shape=tuple(mesh.shape))
        ).float()  # [1, tp, max_seq / tp, W]
        local = cache_sr.shape[2] // sp
        flat = torch.cat([cache_sr[0, t, s * local : (s + 1) * local] for s in range(sp) for t in range(tp)], dim=0)
        p = blockcyclic_positions(sp * tp, self.chunk, self.max_seq)
        nat = torch.empty_like(flat)
        nat[p] = flat
        return nat[:length]


class TtHy4Attention:
    """One layer's gated sparse MLA. ``setup(chunk, max_seq)`` once per geometry, then
    ``__call__(attn_norm, q_resid, topk, start)`` per chunk (start a multiple of the chunk). Writes the chunk's latent
    rows into the layer's device cache and returns attn_out [1, 1, S/2, hidden/2] fp32 (column split)."""

    def __init__(
        self,
        mesh,
        q_b: torch.Tensor,  # [H * (dn + r), q_lora]
        kv_a: torch.Tensor,  # [lat + r, hidden]
        kv_a_norm: torch.Tensor,  # [lat]
        kv_b: torch.Tensor,  # [H * (dn + dv), lat]
        gate: torch.Tensor,  # [H * dv, hidden]
        o_proj: torch.Tensor,  # [hidden, H * dv]
        sink: torch.Tensor,  # [H]
        *,
        n_heads: int,
        nope_dim: int,
        rope_dim: int,
        v_dim: int,
        kv_lora_rank: int,
        rope_theta: float,
        eps: float,
        scale: float,
        sp_axis: int = 0,
        tp_axis: int = 1,
    ):
        assert sp_axis == 0 and tp_axis == 1, "block-cyclic cache / full-mesh gather assume sp_axis 0, tp_axis 1"
        self.mesh, self.sp_axis, self.tp_axis = mesh, sp_axis, tp_axis
        self.sp, self.tp = mesh.shape[sp_axis], mesh.shape[tp_axis]
        assert self.tp > 1, "the deduped latent cache needs a TP axis"
        hq, dn, r, dv, lat = n_heads, nope_dim, rope_dim, v_dim, kv_lora_rank
        hidden = kv_a.shape[1]
        assert hq % self.tp == 0 and (hq // self.tp) % 32 == 0, "sparse_sdpa needs a multiple of 32 heads per chip"
        assert q_b.shape[0] == hq * (dn + r) and kv_a.shape[0] == lat + r and kv_b.shape == (hq * (dn + dv), lat)
        assert gate.shape == (hq * dv, hidden) and o_proj.shape == (hidden, hq * dv) and sink.numel() == hq
        self.n_heads, self.heads_local = hq, hq // self.tp
        self.nope_dim, self.rope_dim, self.v_dim, self.lat = dn, r, dv, lat
        self.kv_width = lat + r
        self.rope_theta, self.eps = float(rope_theta), float(eps)
        # sparse_sdpa's noconvert float caster refuses a double it cannot hold exactly (known issues); 1/16 is exact.
        self.scale = float(torch.tensor(scale, dtype=torch.float32))
        self.num_links = 2 if mesh.arch() == ttnn.Arch.BLACKHOLE else 1  # as ttMLA.ccl_num_links
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        dram = ttnn.DRAM_MEMORY_CONFIG

        def put(t, dtype, dims=None, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh)
                if dims is None
                else ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)
            )
            return ttnn.from_torch(
                t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=mapper
            )

        q_lora = q_b.shape[1]
        # Column (head) split: chip column c holds heads 32c .. 32c + 31 (contiguous output columns of q_b / gate,
        # dim 1 of the per-head kv_b tensors, contiguous K rows of o_proj^T), replicated over the SP rows.
        self.q_b = put(q_b.float().t().reshape(1, 1, q_lora, hq * (dn + r)), ttnn.bfloat16, (None, 3))
        kvb = kv_b.float().reshape(1, hq, dn + dv, lat)
        self.w_uk = put(kvb[:, :, :dn, :], ttnn.bfloat16, (None, 1))  # [1, H, 192, 512]: q_nope -> latent
        self.w_uv = put(kvb[:, :, dn:, :].transpose(-2, -1), ttnn.bfloat16, (None, 1))  # [1, H, 512, 256]
        self.gate = put(gate.float().t().reshape(1, 1, hidden, hq * dv), ttnn.bfloat16, (None, 3))
        self.o_proj = put(o_proj.float().t().reshape(1, 1, hq * dv, hidden), ttnn.bfloat16, (None, 2))
        # kv_a K-split: W^T [hidden, 576], rows split over the TP axis (matches attn_norm's column split).
        self.kv_a = put(kv_a.float().t().reshape(1, 1, hidden, lat + r), ttnn.bfloat16, (None, 2))
        self.kv_a_norm = put(
            kv_a_norm.float().reshape(1, 1, lat // TILE, TILE), ttnn.float32, None, ttnn.ROW_MAJOR_LAYOUT
        )
        # sparse_sdpa multiplies the sink by the scale: pass sink / scale = sink * 16 (exact), [1, 1, 1, H/tp] bf16.
        self.sink = put(
            (sink.float() / self.scale).reshape(1, 1, 1, hq), ttnn.bfloat16, (None, 3), ttnn.ROW_MAJOR_LAYOUT
        )
        self._geoms: dict = {}
        self.geom: _Geometry | None = None
        self.slot = None  # serving binding (bind_cache); None = the geometry's own single-sequence cache

    # ---- load time
    def bind_cache(self, cache, slot: int, row: int, rows: int) -> None:
        """Serving option: write / read this layer's latent rows in an external multi-slot cache (the prefill
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
        """Harness boundary: natural-order kv_latent [n, 576] into the current geometry's cache (zeros past n)."""
        self.geom.load(self.mesh, prefix, self.sp, self.tp, self.kv_width)

    def read_state(self, length: int) -> torch.Tensor:
        """Harness boundary: the latent cache [length, 576] in natural order."""
        return self.geom.read(self.mesh, length, self.sp, self.tp)

    # ---- forward (device only)
    def _rope(self, x, start):
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            self.geom.cos,
            self.geom.sin,
            self.geom.trans,
            start,
            self.sp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ckc,
        )

    def _kv_stem(self, x, start: int) -> None:
        """kv = [kv_a_layernorm(latent) | RoPE(k_rope)] for this chunk -> the latent cache (this chip's window)."""
        dram, s_loc = ttnn.DRAM_MEMORY_CONFIG, x.shape[2]
        part = ttnn.linear(x, self.kv_a, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        kv = ttnn.all_reduce(part, cluster_axis=self.tp_axis, memory_config=dram)  # [1, 1, S/2, 576] fp32
        ttnn.deallocate(part)
        nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, s_loc, self.lat], memory_config=dram)
        rope = ttnn.slice(kv, [0, 0, 0, self.lat], [1, 1, s_loc, self.kv_width], memory_config=dram)
        ttnn.deallocate(kv)
        nn_ = ttnn.bringup.rms_norm(
            nope, weight=self.kv_a_norm, epsilon=self.eps, memory_config=dram, compute_kernel_config=self.ckc
        )
        ttnn.deallocate(nope)
        nb = ttnn.typecast(nn_, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(nn_)
        rb = ttnn.typecast(rope, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(rope)
        rr = self._rope(rb, start)
        ttnn.deallocate(rb)
        kvpe = ttnn.concat([nb, rr], dim=-1, memory_config=dram)
        ttnn.deallocate(nb)
        ttnn.deallocate(rr)
        kvpe_rm = ttnn.to_layout(kvpe, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        ttnn.deallocate(kvpe)
        cache, slot, row, rows = self._cache()
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            kvpe_rm,
            slot_idx=slot,
            layer_idx=row,
            num_layers=rows,
            kv_actual_global=start,
            cluster_axis=self.sp_axis,
            tp_axis=self.tp_axis,
        )
        ttnn.deallocate(kvpe_rm)

    def _q_stem(self, qr, start: int):
        """Absorbed q [1, 32, S/2, 576] bf16: q_b_proj -> heads -> q_nope @ W_uk | RoPE(q_rope)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        q = ttnn.linear(qr, self.q_b, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        q_nope, q_rope = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.heads_local, split_head_dim=self.nope_dim, memory_config=dram
        )
        ttnn.deallocate(q)
        q_lat = ttnn.linear(q_nope, self.w_uk, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(q_nope)
        q_rr = self._rope(q_rope, start)
        ttnn.deallocate(q_rope)
        q_abs = ttnn.concat([q_lat, q_rr], dim=-1, memory_config=dram)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rr)
        return q_abs

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, topk: ttnn.Tensor, start: int) -> ttnn.Tensor:
        dram, g = ttnn.DRAM_MEMORY_CONFIG, self.geom
        assert g is not None, "call setup(chunk, max_seq) first"
        assert start % g.chunk == 0, f"start {start} must be chunk ({g.chunk}) aligned"
        end = start + g.chunk
        self._kv_stem(x, start)
        q_abs = self._q_stem(qr, start)

        # The populated prefix [0, end) of the block-cyclic cache (whole slabs), from all 4 chips into one
        # replicated scratch (one snake over both axes: FABRIC_2D).
        cache, slot, row, rows = self._cache()
        kv_all = ttnn.experimental.high_bw_all_gather(
            cache,
            dim=2,
            output_tensor=g.kv_all,
            num_links=self.num_links,
            cluster_axis=None,
            input_batch_index=slot * rows + row,
            gathered_dim_size=min(g.max_seq, -(-end // g.chunk) * g.chunk),
        )
        q_rm = ttnn.to_layout(q_abs, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)  # the op is ROW_MAJOR only
        ttnn.deallocate(q_abs)
        o = ttnn.transformer.sparse_sdpa(
            q_rm,
            kv_all,
            topk,
            v_dim=self.lat,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=SPARSE_K_CHUNK,
            compute_kernel_config=self.ckc,
            block_cyclic_sp_axis=self.sp_axis,
            block_cyclic_chunk_local=g.chunk_local,
            block_cyclic_cache_tp_sharded=True,
            attention_sink=self.sink,
        )  # [1, 32, S/2, 512] bf16 ROW_MAJOR
        ttnn.deallocate(q_rm)
        o_t = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=dram)
        ttnn.deallocate(o)
        v = ttnn.linear(o_t, self.w_uv, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(o_t)  # [1, 32, S/2, 256]
        vc = ttnn.experimental.nlp_concat_heads(v, memory_config=dram)  # [1, 1, S/2, 32 * 256]
        ttnn.deallocate(v)

        # Output gate, elementwise per (head, v-dim): the row's full attn_norm, head-split linear_gate columns.
        xf = ttnn.all_gather(x, dim=3, cluster_axis=self.tp_axis, memory_config=dram)  # [1, 1, S/2, hidden]
        gl = ttnn.linear(xf, self.gate, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(xf)
        gs = ttnn.sigmoid(gl, memory_config=dram)
        ttnn.deallocate(gl)
        vg = ttnn.multiply(vc, gs, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(vc)
        ttnn.deallocate(gs)

        # o_proj row-parallel (K = this chip's 32 heads x 256), fp32 partials, reduce-scatter to the column split.
        part = ttnn.linear(vg, self.o_proj, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(vg)
        out = ttnn.reduce_scatter(part, dim=3, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(part)
        return out  # [1, 1, S/2, hidden / tp] fp32
