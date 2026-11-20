# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Absorbed sparse MLA.

Adapted from models/demos/deepseek_v3_d_p/tt/mla/mla.py (ttMLA sparse path: _kv_stem, _q_stem, _sparse_mla,
_o_proj_epilogue) without its q-LoRA, its TP-deduped cache and its full-mesh (2D fabric) gather. Per chip (SP row r
of 8, TP column t of 4), for one 5120-token chunk at ``start``:

    x        [1, 1, 640, 1536] attn_norm -> all_gather(axis 1) -> x_full [1, 1, 640, 6144]
    queries  q = x_full @ wq (this column's 16 heads, column-parallel) [1, 1, 640, 16 * 128], times the softmax
             scaling f[p, h] = s_h log(p + 1) + b_h (the reference scales q_nope and q_rot by it; it commutes with
             RoPE and the absorption). f comes from a per-layer table through rotary_embedding_indexed (cos = f,
             sin = 0) applied to ones [640, 32] (one column per local head), widened to 128 columns per head by a
             constant 0/1 matmul. -> heads: q_nope | q_rot [1, 16, 640, 64] each; q_lat = q_nope @ W_uk (per head,
             64 -> 512); q = [q_lat | RoPE(q_rot)] [1, 16, 640, 576] (indexed RoPE with q_lat as concat_prefix).
    keys     kv = x_full @ wkv_a (replicated) [640, 576] -> latent rms_norm (kv_a_norm, eps 1e-5) | RoPE(k_rot) ->
             bfp8 -> update_padded_kv_cache into the served kvpe cache (SP-sharded, TP-replicated, block-cyclic).
    gather   high_bw_all_gather over SP (axis 0) of this slot's cache rows into a persistent bfp8 scratch
             [1, 1, max_seq, 576] (rank-major, i.e. block-cyclic physical order) -> bf16 ROW_MAJOR; this chunk's rows
             are then overwritten with their bf16 values (ATTN_CHUNK_KV: attended before the cache's rounding).
    reshard  sparse_sdpa wants >= 32 heads per chip: heads -> sequence over TP (all_to_all_async_generic, or
             all_gather + mesh_partition, ATTN_RESHARD) -> [1, 64, 160, 576]: chip (r, t) holds query rows
             640 r + 160 t + [0, 160), the rows of the indexer's top-k on that chip.
    attend   sparse_sdpa(scale 128^-0.5, v 512, block-cyclic remap of the natural indices) -> [1, 64, 160, 512]
             -> back to heads [1, 16, 640, 512] -> @ W_uv (512 -> 128) -> concat heads [640, 2048] -> @ wo (rows of
             this column's heads, fp32 partial) -> reduce_scatter(axis 1) -> bf16 [1, 1, 640, 1536].

NoPE layers (rope_dim 0): q is 128 NoPE dims per head, split
into heads by reshape + permute; wkv_a gives only the 512 latent; the kvpe record stays 576 wide with the pe columns
512..575 exact zeros (the latent padded by ttnn.pad), and q = [q_lat | zeros], so the scores are the latent's alone.

Shared-KV layers (kv_source_layers[i] != i): no wkv_a / kv_a_norm and no cache write.
The module reads its source layer's kvpe region (``cache`` / ``cache_layer`` of the source, which has already written
this chunk's rows) through the same SP gather; with ``kvpe`` (the source's bf16 rows of this chunk) those rows are
pasted as on the owning layer (ATTN_CHUNK_KV).
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import mm_configs, seqbuf
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS, TP_AXIS
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.rope import (
    IndexedTables,
    block_cyclic_reorder,
    log_rows,
    log_split,
    rope_tables,
    select_split,
)
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

SENTINEL = -1  # a masked top-k slot; as uint32 it is sparse_sdpa's 0xFFFFFFFF


def _compute_config(mesh, fidelity: str, fp32: bool, l1_acc: bool = False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=l1_acc,
    )


def _topology():
    return getattr(ttnn.Topology, settings.get("ATTN_TOPOLOGY"))


class _HeadTables(IndexedTables):
    """IndexedTables whose columns are also split over TP: each column's chip gets its own 32 table columns."""

    def _put(self, t: torch.Tensor) -> ttnn.Tensor:
        t = block_cyclic_reorder(t.float().reshape(1, 1, *t.shape), self.chunk_local, self.mesh.shape[SP_AXIS])
        dims = [None, None]
        dims[SP_AXIS], dims[TP_AXIS] = 2, 3
        return ttnn.from_torch(
            t,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=dims),
        )


class TtAttention:
    # read at call time, not build time: settings may change between runs of one loaded model
    @property
    def composed(self):
        core = settings.get("ATTN_SDPA")
        return core == "composed" or (core == "auto" and not self.rope_dim)

    @property
    def mm_cfg(self):
        return _compute_config(self.mesh, settings.get("ATTN_MATH_FIDELITY"), settings.get("ATTN_FP32_ACC"))

    @property
    def lin_cfg(self):  # the projections' matmuls (MM_L1_ACC); the rope / norm ops keep mm_cfg
        return _compute_config(
            self.mesh, settings.get("ATTN_MATH_FIDELITY"), settings.get("ATTN_FP32_ACC"), settings.get("MM_L1_ACC")
        )

    @property
    def sdpa_cfg(self):
        return _compute_config(self.mesh, settings.get("ATTN_SDPA_FIDELITY"), settings.get("ATTN_FP32_ACC"))

    def __init__(
        self,
        mesh,
        w: dict,
        *,
        inv_freq: torch.Tensor | None,
        n_heads: int,
        nope_dim: int,
        rope_dim: int,
        v_dim: int,
        kv_lora: int,
        eps: float,
        max_seq: int,
        chunk: int,
        num_users: int = 2,
        cache=None,
        cache_layer: int = 0,
        cache_layers: int = 1,
        pe_dim: int | None = None,
    ):
        """w: wq [Hq*(nope+rope), H], wkv_a [lora+rope, H], kv_a_norm [lora], wkv_b [Hq*(nope+v), lora], wo [H, Hq*v],
        ssmax_s / _b [Hq] (host, any float dtype). NoPE layers: rope_dim 0, wkv_a [lora, H], inv_freq None,
        pe_dim the record's zero pe columns (default rope_dim). cache: the served kvpe cache [num_users * cache_layers, 1,
        max_seq / 8, lora + rope] (allocated here when None); this layer owns its slot ``cache_layer``. Without wkv_a
        (a shared-KV layer) the module only reads slot ``cache_layer``, which its source layer writes."""
        self.mesh, self.n_heads, self.nope, self.rope_dim = mesh, n_heads, nope_dim, rope_dim
        self.v_dim, self.lora = v_dim, kv_lora
        self.eps = eps
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.heads = n_heads // self.tp  # this column's heads
        assert self.heads <= ttnn.TILE_SIZE, f"{self.heads} heads per chip do not fit one ssmax table tile"
        self.max_seq, self.chunk = max_seq, chunk
        if settings.get("ATTN_KV_ROWS") == "extent" and "gathered_prefix_tensor" not in (ttnn.untilize.__doc__ or ""):
            raise RuntimeError(
                "MISTRAL_LARGE_4_ATTN_KV_ROWS=extent needs ttnn.untilize(gathered_prefix_tensor=...), which this "
                "tt-metal does not have; leave ATTN_KV_ROWS at max_seq"
            )
        self.rows = chunk // self.sp  # this SP row's rows per chunk
        # NoPE layers keep the 576-wide record: pe columns written as zeros
        self.pe = rope_dim if pe_dim is None else pe_dim
        self.width = kv_lora + self.pe
        self.scale = (nope_dim + rope_dim) ** -0.5  # explicit: sparse_sdpa's default is K_DIM^-0.5 (576)
        self.cache_layer, self.cache_layers = cache_layer, cache_layers
        self.owns_kv = "wkv_a" in w
        # a KV source layer keeps its chunk's bf16 kvpe rows here for the shared-KV layers that read it (their
        # ``kvpe`` argument), until its next call; set by the model
        self.keep_chunk_kv, self.chunk_kv, self.chunk_kv_start = False, None, None
        self.mirror_layer = None  # a shared-KV layer's own kvpe region (KV_MIRRORS), set by the model
        wdt = ttnn.bfloat8_b
        shape = tuple(mesh.shape)

        def put(t, dtype=wdt, dims=None, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh)
                if dims is None
                else ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=dims)
            )
            return ttnn.from_torch(
                t.float().contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        hd = nope_dim + rope_dim
        on_tp = lambda d: (d, None) if TP_AXIS == 0 else (None, d)  # noqa: E731  (tensor dim d split over TP)
        # Weights stay [N, K] (checkpoint order) and are read with transpose_b, so a bfp8 block (16 values of a tile
        # row) runs along the contraction; [K, N] blocks would share exponents across output columns of very
        # different norms.
        # a shared-KV layer attends its source's bfp8 prefix: its own weights in ATTN_SHARED_KV_WEIGHTS
        own_dt = wdt if self.owns_kv else getattr(ttnn, settings.get("ATTN_SHARED_KV_WEIGHTS"))
        self.wq = put(w["wq"].float()[None, None], dtype=own_dt, dims=on_tp(2))  # [heads * hd, H]: this column's heads
        if self.owns_kv:
            # [lora + rope, H], replicated; ATTN_WKV_A_DTYPE: its bfp8 error flattens the softmax over the prefix
            self.wkv_a = put(w["wkv_a"].float()[None, None], dtype=getattr(ttnn, settings.get("ATTN_WKV_A_DTYPE")))
            self.kv_norm_w = put(
                w["kv_a_norm"].float().reshape(1, 1, kv_lora // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        kv_b = w["wkv_b"].float().view(n_heads, nope_dim + v_dim, kv_lora)
        # the absorbed wkv_b halves, bfp8 blocks along the latent 512 (the contraction)
        kvb_dt = getattr(ttnn, settings.get("ATTN_WKV_B_DTYPE")) if self.owns_kv else own_dt
        self.w_uk = put(kv_b[:, :nope_dim][None], dtype=kvb_dt, dims=on_tp(1))  # [1, heads, nope, lora]
        self.w_uv = put(kv_b[:, nope_dim:][None], dtype=kvb_dt, dims=on_tp(1))  # [1, heads, v, lora]
        self.wo = put(w["wo"].float()[None, None], dims=on_tp(3))  # [H, heads * v]: this column's heads

        # the host inputs of the max_seq-sized tables, kept for resize
        self._inv_freq, self._ssmax_sb = inv_freq, (w["ssmax_s"].float().clone(), w["ssmax_b"].float().clone())
        self.ones = put(torch.ones(1, 1, self.rows, ttnn.TILE_SIZE), dtype=ttnn.bfloat16)
        widen = torch.zeros(1, 1, ttnn.TILE_SIZE, self.heads * hd)  # f per head -> its hd columns of q
        for h in range(self.heads):
            widen[..., h, h * hd : (h + 1) * hd] = 1
        self.widen = put(widen, dtype=ttnn.bfloat16)

        self.num_users, self._own_cache = num_users, cache is None
        self.cache = cache
        self.rope = self.ssmax = self.ssmax_sb = self.kv_full = self._log = None
        self.build_seq(max_seq, cache)

    # ---- the max_seq-sized tensors (rebuilt by resize, shared over layers with SEQ_BUFFERS=shared)
    def build_seq(self, max_seq: int, cache=None) -> None:
        """RoPE and softmax-scaling tables, the gather scratch and (allocated here when None) the cache, for max_seq."""
        mesh, chunk = self.mesh, self.chunk
        self.max_seq = max_seq
        self.rope = rope_tables(mesh, self._inv_freq, max_seq, chunk) if self.rope_dim else None
        # softmax scaling f[p, h] = s_h log(p + 1) + b_h, fp32 on the host as the reference; column block t (32
        # wide) holds column t's heads, then zeros
        s, b = self._ssmax_sb
        self.ssmax = self.ssmax_sb = self._log = None
        if settings.get("SSMAX_TABLE") == "shared_log":
            # f = s * log(p + 1) + b in fp32 on the device from one log table per mesh (eager: RowTables slice;
            # traced: SplitTables selection at meta.start), rounded to bf16; this column's s_h, b_h as rows
            # [1, 1, 1, 32] (zeros past its heads)
            rows_sb = []
            for v in (s, b):
                r = torch.zeros(1, 1, 1, self.tp * ttnn.TILE_SIZE)
                for t in range(self.tp):
                    r[..., t * ttnn.TILE_SIZE : t * ttnn.TILE_SIZE + self.heads] = v[
                        t * self.heads : (t + 1) * self.heads
                    ]
                dims = [None, None]
                dims[TP_AXIS] = 3
                rows_sb.append(
                    ttnn.from_torch(
                        r,
                        device=mesh,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
                    )
                )
            self.ssmax_sb = tuple(rows_sb)
        else:
            f = s[None, :] * torch.arange(1, max_seq + 1).float().log()[:, None] + b[None, :]
            tab = torch.zeros(max_seq, self.tp * ttnn.TILE_SIZE)
            for t in range(self.tp):
                tab[:, t * ttnn.TILE_SIZE : t * ttnn.TILE_SIZE + self.heads] = f[
                    :, t * self.heads : (t + 1) * self.heads
                ]
            del f
            zero = seqbuf.get(
                mesh,
                ("ssmax_sin", max_seq, chunk, tab.shape[1]),
                lambda: _HeadTables.table(mesh, torch.zeros_like(tab), chunk),
            )
            self.ssmax = _HeadTables(mesh, tab, zero, chunk)
            del tab
        self.cache = cache if cache is not None else self.cache
        if self.cache is None:
            self.cache = init_kvpe_cache(
                self.width,
                mesh,
                max_seq,
                tuple(mesh.shape),
                SP_AXIS,
                self.cache_layers,
                num_users=self.num_users,
                dtype=kv_cache_storage()[0],
                layout=kv_cache_storage()[1],
            )
            _stamp_sp_topology(self.cache, mesh)
        # the SP gather's persistent output: one slot's rows of every SP row, rank-major (block-cyclic physical)
        dt, lay = self.cache.dtype, self.cache.layout
        self.kv_full = seqbuf.get(
            mesh,
            ("kv_full", max_seq, self.width, str(dt), str(lay)),
            lambda: ttnn.from_torch(
                torch.zeros(1, 1, max_seq, self.width),
                device=mesh,
                dtype=dt,
                layout=lay,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            ),
        )

    def release_seq(self) -> None:
        """Frees this layer's max_seq-sized tensors (a shared one stays: seqbuf.release); its own cache too."""
        for obj in (self.rope, self.ssmax, self.ssmax_sb, self.kv_full, self._log):
            seqbuf.free(obj)
        self.rope = self.ssmax = self.ssmax_sb = self.kv_full = self._log = None
        if self._own_cache and self.cache is not None:
            ttnn.deallocate(self.cache)
            self.cache = None
        if self.chunk_kv is not None:
            ttnn.deallocate(self.chunk_kv)
        self.chunk_kv = self.chunk_kv_start = None

    def _scaling(self, start: int, meta=None) -> ttnn.Tensor:
        """f[p, h] of this chip's rows and heads, bf16 [1, 1, 640, 32] (zeros past its heads)."""
        if self.ssmax_sb is None:
            return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                self.ones,
                self.ssmax.cos,
                self.ssmax.sin,
                self.ssmax.trans,
                start if meta is None else meta.start,
                cluster_axis=SP_AXIS,
                compute_kernel_config=self.mm_cfg,
            )  # cos = f, sin = 0
        if meta is None:
            if self._log is None:  # the eager path's fp32 log(p + 1) rows, built on first eager use
                self._log = log_rows(self.mesh, self.max_seq, self.chunk, self.rows)
            lg = self._log.rows(start)
        else:
            # built and prepared on the first metadata-path chunk, which runs eagerly before any capture
            t = log_split(self.mesh, self.max_seq, self.chunk)
            t.prepare(self.rows)
            lg = select_split(meta, t, self.rows, False)
        fs = ttnn.multiply(lg, self.ssmax_sb[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if meta is None and start % self.chunk:
            ttnn.deallocate(lg)  # a selection of its own (a chunk-aligned start's is a slice of the table)
        f32 = ttnn.add(fs, self.ssmax_sb[1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(fs)
        f = ttnn.typecast(f32, ttnn.bfloat16)
        ttnn.deallocate(f32)
        return f

    # ---- pieces
    def _rope(self, x, start: int, prefix=None, meta=None):
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            self.rope.cos,
            self.rope.sin,
            self.rope.trans,
            start if meta is None else meta.start,
            cluster_axis=SP_AXIS,
            rotary_dim=self.rope_dim,
            concat_prefix=prefix,
            compute_kernel_config=self.mm_cfg,
        )

    def _linear(self, x, w, transpose_b=True, dtype=None, name=None):
        return ttnn.matmul(
            x,
            w,
            transpose_b=transpose_b,
            dtype=dtype,
            program_config=mm_configs.get(name, x, w, transpose_b) if name else None,
            compute_kernel_config=self.lin_cfg if name else self.mm_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def write_kv(self, x_full, start: int, slot: int, end: int, meta=None):
        """kvpe rows of this chunk -> the cache rows [start, end) of (slot, cache_layer). Returns them in bf16.
        meta (chunk_meta.py): slot, start and end read on the device instead (trace-safe forms)."""
        kv = self._linear(
            x_full, self.wkv_a, dtype=ttnn.bfloat16, name="wkv_a"
        )  # [1, 1, 640, 576] ([1, 1, 640, 512] on NoPE layers)
        if self.rope_dim:
            lat = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, self.rows, self.lora])
            rot = ttnn.slice(kv, [0, 0, 0, self.lora], [1, 1, self.rows, self.width])
            ttnn.deallocate(kv)
        else:
            lat, rot = kv, None
        latn = ttnn.rms_norm(
            lat,
            weight=self.kv_norm_w,
            epsilon=self.eps,
            compute_kernel_config=self.mm_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(lat)
        if self.rope_dim:
            kvpe = self._rope(rot, start, prefix=latn, meta=meta)  # [latent | RoPE(k_rot)]
            ttnn.deallocate(rot)
        else:
            kvpe = self._pad_pe(latn)  # [latent | 0]
        ttnn.deallocate(latn)
        self.write_rows(kvpe, self.cache_layer, start, slot, end, meta=meta)
        return kvpe

    def write_rows(self, kvpe, layer: int, start: int, slot: int, end: int, meta=None) -> None:
        """bf16 TILE kvpe rows of this chunk -> cache region ``layer`` at [start, end). A ROW_MAJOR cache keeps its
        pad rows as written (zero_padded_kv_cache takes TILE caches only), as the DSA models' bf16 caches do."""
        if self.cache.layout == ttnn.ROW_MAJOR_LAYOUT:
            kc = ttnn.to_layout(kvpe, ttnn.ROW_MAJOR_LAYOUT)
        else:
            kc = ttnn.typecast(kvpe, self.cache.dtype)
        tiled = self.cache.layout == ttnn.TILE_LAYOUT
        if meta is not None:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.cache,
                kc,
                meta.slot,
                meta.start,
                layer_idx=layer,
                num_layers=self.cache_layers,
                cluster_axis=SP_AXIS,
                valid_global=meta.end,
            )
            if tiled:
                ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                    self.cache,
                    meta.slot,
                    meta.end,
                    layer,
                    self.cache_layers,
                    self.chunk,
                    SP_AXIS,
                    pad_align=ttnn.TILE_SIZE,
                )
            ttnn.deallocate(kc)
            return
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            self.cache,
            kc,
            slot_idx=slot,
            layer_idx=layer,
            num_layers=self.cache_layers,
            kv_actual_global=start,
            cluster_axis=SP_AXIS,
            valid_global=end,
        )
        ttnn.deallocate(kc)
        if tiled and end < start + self.chunk and end % ttnn.TILE_SIZE:
            ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                self.cache,
                slot,
                layer,
                self.cache_layers,
                end,
                self.chunk,
                SP_AXIS,
                pad_align=ttnn.TILE_SIZE,
            )

    def _pad_pe(self, t):
        """[..., lora] -> [..., lora + pe], the pe columns exact zeros (NoPE layers)."""
        return ttnn.pad(t, [(0, 0), (0, 0), (0, 0), (0, self.pe)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _queries(self, x_full, start: int, meta=None):
        q = self._linear(x_full, self.wq, dtype=ttnn.bfloat16, name="wq")  # [1, 1, 640, heads * 128]
        f = self._scaling(start, meta)  # f[p, h] of this chip's rows and heads
        fw = self._linear(f, self.widen, transpose_b=False)  # [1, 1, 640, heads * 128], exact (one term per column)
        ttnn.deallocate(f)
        qs = ttnn.multiply(q, fw, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q)
        ttnn.deallocate(fw)
        if self.rope_dim:
            q_nope, q_rot = ttnn.experimental.nlp_create_q_heads_split(
                qs, num_heads=self.heads, split_head_dim=self.nope, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # [1, heads, 640, 64] each
            ttnn.deallocate(qs)
        else:  # all 128 dims are NoPE: [1, 1, 640, heads * 128] -> [1, heads, 640, 128]
            q4 = ttnn.reshape(qs, [1, self.rows, self.heads, self.nope])
            ttnn.deallocate(qs)
            q_nope = ttnn.permute(q4, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q4)
        q_lat = self._linear(q_nope, self.w_uk, transpose_b=False, name="w_uk")  # [1, heads, 640, 512]
        ttnn.deallocate(q_nope)
        if not self.rope_dim:
            qq = self._pad_pe(q_lat)  # [1, heads, 640, 576], q_rot = 0
            ttnn.deallocate(q_lat)
            return qq
        qq = self._rope(q_rot, start, prefix=q_lat, meta=meta)  # [1, heads, 640, 576]
        ttnn.deallocate(q_rot)
        ttnn.deallocate(q_lat)
        return qq

    def _to_seq(self, t):
        """[1, heads, 640, W] per chip (heads over TP) -> [1, Hq, 160, W] (query rows over TP)."""
        if settings.get("ATTN_RESHARD") == "all_to_all":
            return ttnn.experimental.all_to_all_async_generic(
                t,
                in_dim=1,
                out_dim=2,
                num_links=settings.get("NUM_LINKS"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=TP_AXIS,
            )
        full = ttnn.all_gather(
            t, dim=1, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=_topology()
        )
        out = ttnn.mesh_partition(full, dim=2, cluster_axis=TP_AXIS)
        ttnn.deallocate(full)
        return out

    def _to_heads(self, t):
        """Inverse of _to_seq."""
        if settings.get("ATTN_RESHARD") == "all_to_all":
            return ttnn.experimental.all_to_all_async_generic(
                t,
                in_dim=2,
                out_dim=1,
                num_links=settings.get("NUM_LINKS"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=TP_AXIS,
            )
        full = ttnn.all_gather(
            t, dim=2, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=_topology()
        )
        out = ttnn.mesh_partition(full, dim=1, cluster_axis=TP_AXIS)
        ttnn.deallocate(full)
        return out

    def _gather_kv(self, slot: int, end: int, meta=None) -> ttnn.Tensor:
        """This slot's kvpe rows of every SP row -> bf16 ROW_MAJOR [1, 1, max_seq, 576], rank-major. meta: the slot
        and the gathered extent (min(round_up(start + chunk, chunk), max_seq), which covers ceil(end / chunk) slabs)
        read on the device."""
        slab = self.chunk  # one block of chunk/8 rows on each SP row
        if meta is None:
            where = dict(
                input_batch_index=slot * self.cache_layers + self.cache_layer,
                gathered_dim_size=min(self.max_seq, -(-end // slab) * slab),
            )
        else:
            where = dict(
                input_batch_index_tensor=meta.slot,
                batch_slot_num_layers=self.cache_layers,
                batch_slot_layer_idx=self.cache_layer,
                gathered_prefix_tensor=meta.start,
                gathered_slab_global=slab,
            )
        gathered = ttnn.experimental.high_bw_all_gather(
            self.cache,
            dim=2,
            output_tensor=self.kv_full,
            cluster_axis=SP_AXIS,
            num_links=settings.get("NUM_LINKS"),
            **where,
        )
        if gathered.layout == ttnn.ROW_MAJOR_LAYOUT:
            # the op returns a new handle on kv_full's buffer; hand back kv_full itself so callers, which free only
            # what is not self.kv_full, keep the persistent scratch
            return self.kv_full
        g = min(self.max_seq, -(-end // slab) * slab)
        if (
            meta is None
            and settings.get("ATTN_KV_ROWS") == "gathered"
            and g < self.max_seq
            and gathered.dtype == ttnn.bfloat8_b
        ):
            # only the gathered slabs (g / 8 rows of each rank's max_seq / 8) -> bf16 ROW_MAJOR [1, 1, g, 576], still
            # rank-major: sparse_sdpa's block-cyclic remap takes its per-rank stride from the kv rows (T / sp)
            v = ttnn.reshape(gathered, [1, self.sp, self.max_seq // self.sp, self.width])
            rm = ttnn.untilize_with_unpadding(
                v,
                output_tensor_end=[0, self.sp - 1, g // self.sp - 1, self.width - 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.reshape(rm, [1, 1, g, self.width])
        if meta is not None and settings.get("ATTN_KV_ROWS") == "extent":
            # untilize only the slabs the gather wrote (g = min(round_up(start + chunk, chunk), max_seq)), the extent
            # read on the device so one capture serves every chunk; rows past g stay stale (sparse_sdpa reads only
            # top-k positions < end <= g)
            return ttnn.untilize(
                gathered,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                gathered_prefix_tensor=meta.start,
                gathered_slab_global=slab,
                num_segments=self.sp,
            )
        if gathered.dtype in (ttnn.bfloat8_b, ttnn.bfloat16) and settings.get("ATTN_KV_UNTILIZE") == "fused":
            # untilize unpacks bfp8 to bf16 itself (exact), no separate typecast
            return ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        kv16 = ttnn.typecast(gathered, ttnn.bfloat16)
        kv_rm = ttnn.to_layout(kv16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(kv16)
        return kv_rm

    def _gather_chunk_rm(self, kvpe):
        """The chunk's bf16 kvpe rows of every SP row, ROW_MAJOR [1, 1, chunk, 576] (ATTN_PASTE_GATHER)."""
        links = settings.get("NUM_LINKS")
        if settings.get("ATTN_PASTE_GATHER") == "row_major":
            # untilize this row's rows, then gather (same values, 8x smaller untilize)
            rows_rm = ttnn.to_layout(kvpe, ttnn.ROW_MAJOR_LAYOUT)
            full_rm = ttnn.all_gather(rows_rm, dim=2, cluster_axis=SP_AXIS, num_links=links, topology=_topology())
            ttnn.deallocate(rows_rm)
        else:
            full = ttnn.all_gather(kvpe, dim=2, cluster_axis=SP_AXIS, num_links=links, topology=_topology())
            full_rm = ttnn.to_layout(full, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, chunk, 576], SP row r at [r rows, +rows)
            ttnn.deallocate(full)
        return full_rm

    def _paste_chunk_meta(self, kv_rm, kvpe, meta) -> None:
        """_paste_chunk at a device-held start: SP row k's ``rows`` keys go to its rank segment of the gathered prefix
        at the offset update_padded_kv_cache gives SP row k for the chunk (one segment = one batch slot of a
        [sp, 1, max_seq / sp, 576] view); every chip writes all sp segments of its replicated copy, so each write
        takes the per-chip start meta.paste[k] whose offset on that chip is SP row k's. The same rows land in the same
        places as the slab and runs paths, for any 32-aligned start."""
        rows, sp, per_rank = self.rows, self.sp, self.max_seq // self.sp
        full_rm = self._gather_chunk_rm(kvpe)  # [1, 1, chunk, 576], SP row k at [k rows, +rows)
        dst = ttnn.reshape(kv_rm, [sp, 1, per_rank, self.width])
        for k in range(sp):
            piece = ttnn.slice(full_rm, [0, 0, k * rows, 0], [1, 1, (k + 1) * rows, self.width])
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                dst, piece, meta.zero, meta.paste[k], layer_idx=k, num_layers=sp, cluster_axis=SP_AXIS
            )
            ttnn.deallocate(piece)
        ttnn.deallocate(full_rm)

    def _paste_chunk(self, kv_rm, kvpe, start: int) -> None:
        """Overwrites this chunk's rows of the gathered (bfp8-rounded) prefix with their bf16 values: the chunk
        attends to its own keys before the cache's rounding, as the reference does; the prefix stays the served bfp8.
        SP row r holds the chunk positions g with (g // rows) % sp == r, ascending (the served rotation); position g
        sits in the rank-major gather at rank (g // rows) % sp, local row (g // chunk) * rows + g % rows."""
        rows, sp = self.rows, self.sp
        per_rank = kv_rm.shape[2] // sp  # max_seq / 8, or g / 8 for the gathered rows only (ATTN_KV_ROWS)
        full_rm = self._gather_chunk_rm(kvpe)
        if settings.get("ATTN_PASTE") == "slab" and start % self.chunk == 0 and start + self.chunk <= kv_rm.shape[2]:
            # a chunk-aligned chunk is one block of ``rows`` per SP row at the same local offset on every rank: one
            # strided write through [1, sp, per_rank, width] views
            c = start // self.chunk
            dst = ttnn.reshape(kv_rm, [1, sp, per_rank, self.width])
            src = ttnn.reshape(full_rm, [1, sp, rows, self.width])
            ttnn.experimental.slice_write(
                src, dst, [0, 0, c * rows, 0], [1, sp, (c + 1) * rows, self.width], [1, 1, 1, 1]
            )
            ttnn.deallocate(full_rm)
            return
        taken = [0] * sp
        g, stop = start, min(start + self.chunk, self.max_seq)
        while g < stop:
            run = min(stop, (g // rows + 1) * rows) - g  # positions of one block of ``rows``
            r = (g // rows) % sp
            src = r * rows + taken[r]
            dst = r * per_rank + (g // self.chunk) * rows + g % rows
            piece = ttnn.slice(full_rm, [0, 0, src, 0], [1, 1, src + run, self.width])
            ttnn.experimental.slice_write(piece, kv_rm, [0, 0, dst, 0], [1, 1, dst + run, self.width], [1, 1, 1, 1])
            ttnn.deallocate(piece)
            taken[r] += run
            g += run
        ttnn.deallocate(full_rm)

    def _attend_composed(self, q_seq, kv_rm, topk):
        """sparse_sdpa composed from stock ops (ATTN_SDPA): per block of ATTN_COMPOSED_ROWS query rows, the top-k keys
        are gathered (embedding) from the prefix put in natural order; fp32 scores and exp(s - max); the bf16 weights
        @ the latent in fp32, divided by those weights' own sum. q_seq [1, Hq, 160, 576] TILE, kv_rm [1, 1, max_seq, 576] rank-major ROW_MAJOR, topk uint32
        [1, 1, 160, k]. Returns [1, Hq, 160, 512] TILE."""
        rows_q, k = topk.shape[2], topk.shape[3]
        hq = q_seq.shape[1]
        # rank-major (block-cyclic physical) -> natural positions: [sp, blocks, rows] -> [blocks, sp, rows]
        n_kv = kv_rm.shape[2]  # max_seq, or the gathered rows only (ATTN_KV_ROWS)
        kv4 = ttnn.reshape(kv_rm, [self.sp, n_kv // (self.sp * self.rows), self.rows, self.width])
        kv_nat = ttnn.reshape(
            ttnn.permute(kv4, (1, 0, 2, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG), [n_kv, self.width]
        )
        # sentinels: index 0, score -1e30 (a row with no valid key stays finite)
        t_tile = ttnn.to_layout(topk, ttnn.TILE_LAYOUT)
        tf = ttnn.typecast(t_tile, ttnn.float32)
        ttnn.deallocate(t_tile)
        valid = ttnn.lt(tf, float(self.max_seq))  # 0xFFFFFFFF -> 4.3e9
        idx_f = ttnn.multiply(tf, valid)
        ttnn.deallocate(tf)
        idx_u = ttnn.typecast(idx_f, ttnn.uint32)
        ttnn.deallocate(idx_f)
        idx = ttnn.to_layout(idx_u, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(idx_u)
        bias = ttnn.multiply(valid, 1e30)
        ttnn.deallocate(valid)
        bias_s = ttnn.subtract(bias, 1e30)  # [1, 1, 160, k] fp32: 0 or -1e30
        ttnn.deallocate(bias)
        bias = bias_s
        q_rows = ttnn.permute(q_seq, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 160, Hq, 576]
        b = settings.get("ATTN_COMPOSED_ROWS")
        outs = []
        for a in range(0, rows_q, b):
            ib = ttnn.reshape(ttnn.slice(idx, [0, 0, a, 0], [1, 1, a + b, k]), [1, b * k])
            kb = ttnn.embedding(ib, kv_nat, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ib)
            kb = ttnn.reshape(kb, [1, b, k, self.width])
            qb = ttnn.slice(q_rows, [0, a, 0, 0], [1, a + b, hq, self.width])
            sc = ttnn.matmul(
                qb,
                kb,
                transpose_b=True,
                dtype=ttnn.float32,
                compute_kernel_config=self.sdpa_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, b, Hq, k]
            ttnn.deallocate(qb)
            mb = ttnn.reshape(ttnn.slice(bias, [0, 0, a, 0], [1, 1, a + b, k]), [1, b, 1, k])
            scs = ttnn.multiply(sc, self.scale)
            ttnn.deallocate(sc)
            sc2 = ttnn.add(scs, mb)
            ttnn.deallocate(scs)
            ttnn.deallocate(mb)
            # unnormalized exp(s - max) in fp32, then bf16; the output is divided by the sum of those bf16 values,
            # so the weights it averages with sum to 1 exactly
            mx = ttnn.max(sc2, dim=-1, keepdim=True)
            d = ttnn.subtract(sc2, mx)
            ttnn.deallocate(sc2)
            ttnn.deallocate(mx)
            p = ttnn.exp(d, fast_and_approximate_mode=False)
            ttnn.deallocate(d)
            pb = ttnn.typecast(p, ttnn.bfloat16)
            ttnn.deallocate(p)
            pf = ttnn.typecast(pb, ttnn.float32)
            rs = ttnn.sum(pf, dim=-1, keepdim=True, compute_kernel_config=self.sdpa_cfg)
            ttnn.deallocate(pf)
            vb = ttnn.slice(kb, [0, 0, 0, 0], [1, b, k, self.lora])
            ttnn.deallocate(kb)
            ob = ttnn.matmul(
                pb,
                vb,
                dtype=ttnn.float32,
                compute_kernel_config=self.sdpa_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(pb)
            ttnn.deallocate(vb)
            inv = ttnn.reciprocal(rs)
            ttnn.deallocate(rs)
            on = ttnn.multiply(ob, inv, dtype=ttnn.bfloat16)
            ttnn.deallocate(ob)
            ttnn.deallocate(inv)
            outs.append(on)  # [1, b, Hq, 512]
        ttnn.deallocate(q_rows)
        ttnn.deallocate(idx)
        ttnn.deallocate(bias)
        ttnn.deallocate(kv_nat)
        o = ttnn.concat(outs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(outs) > 1 else outs[0]
        out = ttnn.permute(o, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, Hq, 160, 512]
        for t in outs:
            ttnn.deallocate(t)
        if len(outs) > 1:
            ttnn.deallocate(o)
        return out

    def __call__(
        self,
        x: ttnn.Tensor,
        topk: ttnn.Tensor,
        start: int,
        slot: int = 0,
        end: int | None = None,
        kvpe: ttnn.Tensor | None = None,
        meta=None,
    ) -> ttnn.Tensor:
        """x: attn_norm [1, 1, S/8, H/4] bf16 or fp32 (ATTN_NORM_DTYPE) per chip, the chunk at absolute positions
        [start, start + S); topk:
        uint32 [1, 1, S/32, k] per chip (chip (r, t): query rows 640 r + 160 t + [0, 160)), natural key positions,
        0xFFFFFFFF pads as a contiguous tail. kvpe (shared-KV layers only, optional): the source layer's bf16 kvpe rows
        of this chunk [1, 1, S/8, 576] per chip, as its write_kv returns them; not deallocated here. Returns attn_out
        [1, 1, S/8, H/4] bf16 per chip. meta (chunk_meta.py): the chunk's slot, start and end read on the device
        (the trace-safe path; start / slot / end are then only the host's view)."""
        end = start + self.chunk if end is None else end
        if x.shape[-1] == self.wq.shape[-1]:  # NORM_GATHER=input: attn_norm handed on the full row
            x_full = x
        else:
            x_full = ttnn.all_gather(
                x, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=_topology()
            )
        if self.owns_kv:
            assert kvpe is None, "a layer that owns its KV writes its own kvpe rows"
            kvpe = self.write_kv(x_full, start, slot, end, meta=meta)
        q = self._queries(x_full, start, meta=meta)
        if x_full is not x:
            ttnn.deallocate(x_full)

        q_seq = self._to_seq(q)  # [1, Hq, 160, 576]
        ttnn.deallocate(q)
        kv = self._gather_kv(slot, end, meta=meta)
        if kvpe is not None and settings.get("ATTN_CHUNK_KV") == "bfloat16" and self.cache.dtype != ttnn.bfloat16:
            if meta is None:
                self._paste_chunk(kv, kvpe, start)
            else:
                self._paste_chunk_meta(kv, kvpe, meta)
        if self.owns_kv and self.keep_chunk_kv:
            if self.chunk_kv is not None:
                ttnn.deallocate(self.chunk_kv)
            self.chunk_kv, self.chunk_kv_start = kvpe, start
        elif self.owns_kv:
            ttnn.deallocate(kvpe)
        if self.composed:
            o_t = self._attend_composed(q_seq, kv, topk)
            ttnn.deallocate(q_seq)
            if kv is not self.kv_full:
                ttnn.deallocate(kv)
            return self._out_proj(o_t)
        q_rm = ttnn.to_layout(q_seq, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_seq)
        k_chunk = settings.get("ATTN_SDPA_K_CHUNK")
        out = ttnn.transformer.sparse_sdpa(
            q_rm,
            kv,
            topk,
            v_dim=self.lora,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=k_chunk,
            compute_kernel_config=self.sdpa_cfg,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=self.rows,
            block_cyclic_cache_tp_sharded=False,
        )  # [1, Hq, 160, 512] ROW_MAJOR
        ttnn.deallocate(q_rm)
        if kv is not self.kv_full:
            ttnn.deallocate(kv)
        o_t = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out)
        return self._out_proj(o_t)

    def _out_proj(self, o_t):
        """[1, Hq, 160, 512] (query rows over TP) -> attn_out [1, 1, 640, 1536] bf16."""
        o_lat = self._to_heads(o_t)  # [1, heads, 640, 512]
        ttnn.deallocate(o_t)
        o = self._linear(o_lat, self.w_uv, name="w_uv")  # [1, heads, 640, 128]
        ttnn.deallocate(o_lat)
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, 640, heads*128]
        ttnn.deallocate(o)
        # [1, 1, 640, H] partial over this column's heads. The 4 partials can cancel to a small sum, so they are summed
        # in ATTN_PARTIAL_DTYPE before rounding to bf16
        part = ttnn.matmul(
            oc,
            self.wo,
            transpose_b=True,
            dtype=getattr(ttnn, settings.get("ATTN_PARTIAL_DTYPE")),
            program_config=mm_configs.get("wo", oc, self.wo),
            compute_kernel_config=self.lin_cfg,
            memory_config=mm_configs.out_memory("wo"),
        )
        ttnn.deallocate(oc)
        y = ttnn.reduce_scatter(
            part, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=_topology()
        )
        ttnn.deallocate(part)
        if y.dtype != ttnn.bfloat16 and not settings.get("RESIDUAL_FP32_IN"):
            yb = ttnn.typecast(y, ttnn.bfloat16)
            ttnn.deallocate(y)
            y = yb
        return y


def kv_cache_storage():
    """(dtype, layout) of the served kvpe cache (KV_CACHE_FORMAT)."""
    if settings.get("KV_CACHE_FORMAT") == "bf16_rm":
        return ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    return ttnn.bfloat8_b, ttnn.TILE_LAYOUT


def _write_cache_meta(cache, rows, meta, layer: int, layers: int, chunk: int) -> None:
    """update_padded_kv_cache of a chunk's rows at the device-held slot / start, clamped to the real tokens (end), then
    the last record's pad rows zeroed (zero_padded_kv_cache: an empty window when end is 32-aligned, as the eager
    path's skip)."""
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        rows,
        meta.slot,
        meta.start,
        layer_idx=layer,
        num_layers=layers,
        cluster_axis=SP_AXIS,
        valid_global=meta.end,
    )
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, meta.slot, meta.end, layer, layers, chunk, SP_AXIS, pad_align=ttnn.TILE_SIZE
    )


def _stamp_sp_topology(cache, mesh) -> None:
    """init_kvpe_cache(tp_axis=None) declares a 1D replicated topology; the cache is really dim 2 sharded over SP
    and replicated over TP, and high_bw_all_gather(cluster_axis=SP) needs that 2D distribution declared."""
    dist = ttnn.MeshShape(mesh.shape[0], mesh.shape[1])
    placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    placements[SP_AXIS] = ttnn.PlacementShard(2)
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    cache.update_tensor_topology(ttnn.TensorTopology(dist, placements, coords))


# ---- host <-> device helpers for tests (never called from a forward)
def topk_to_device(topk: torch.Tensor, mesh) -> ttnn.Tensor:
    """[S, k] host top-k (natural positions, -1 pads) -> uint32 [1, 1, S/32, k] per chip, chip (r, t) holding rows
    (4 r + t) * S/32 + [0, S/32) (the indexer's layout), pads 0xFFFFFFFF."""
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
    t = topk.to(torch.int64)
    t = torch.where(t < 0, torch.full_like(t, SENTINEL), t).to(torch.int32)
    t = t.reshape(sp, tp, -1, t.shape[-1]) if SP_AXIS == 0 else t.reshape(tp, sp, -1, t.shape[-1])
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
    )


def kvpe_cache_load(mod: TtAttention, latent, rope, length: int, mesh, slot: int = 0) -> None:
    """Writes the kvpe rows of positions [0, length) (a golden prefix: latent [>= length, 512], rope [>= length, 64]
    host; [>= length, 0] on NoPE layers, whose pe columns are written as zeros) into the module's served cache, a chunk
    at a time through update_padded_kv_cache, where the module writes."""
    if not length:
        return
    assert latent is not None and rope is not None and latent.shape[0] >= length, "a kvpe prefix is required"
    assert length % mod.chunk == 0, f"prefix {length} is not whole chunks of {mod.chunk}"
    dims = [None, None]
    dims[SP_AXIS] = 2
    for a in range(0, length, mod.chunk):
        rows = torch.cat([latent[a : a + mod.chunk].float(), rope[a : a + mod.chunk].float()], dim=-1)
        rows = torch.nn.functional.pad(rows, (0, mod.width - rows.shape[-1]))  # NoPE: pe columns zero
        k = ttnn.from_torch(
            rows.reshape(1, 1, mod.chunk, -1),
            device=mesh,
            dtype=mod.cache.dtype,
            layout=mod.cache.layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            mod.cache,
            k,
            slot_idx=slot,
            layer_idx=mod.cache_layer,
            num_layers=mod.cache_layers,
            kv_actual_global=a,
            cluster_axis=SP_AXIS,
        )
        ttnn.deallocate(k)


def kvpe_rows_to_device(mod: TtAttention, latent, rope, mesh) -> ttnn.Tensor:
    """A chunk's kvpe rows (latent [S, 512], rope [S, r] host, r 0 on NoPE layers) -> bf16 [1, 1, S/8, 576] per chip
    (rows over SP, replicated over TP), what write_kv returns: a shared-KV layer's source rows of the chunk."""
    rows = torch.cat([latent.float(), rope.float()], dim=-1)
    rows = torch.nn.functional.pad(rows, (0, mod.width - rows.shape[-1]))
    dims = [None, None]
    dims[SP_AXIS] = 2
    return ttnn.from_torch(
        rows.reshape(1, 1, *rows.shape),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
    )
