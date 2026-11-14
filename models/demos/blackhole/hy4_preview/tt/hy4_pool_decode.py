# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context decode of the Hy4 chain on the sequence-sharded SCALED_FP8 pool (task integ, step 1; kv-design/DESIGN.md).

Per decode step and layer (B = 4: one user per mesh row, S = 1 token per user, 8 columns = TP8 head groups AND 8 sequence shards):
    front (Hy4AttentionOpt.front: q / kv / gate / indexer projections, optional HY4_OPT fusions)
    kv row  -> stock per_token_cast_to_fp8 + pack (SCALED_FP8 row 656 B) -> ``row_write`` into the owner column's pool row (all other columns skip)
    full indexer layers: key (layer_norm + rope) -> ``paged_update_cache`` into the bfp8 key pool (owner column; -1 elsewhere)
        topk mode: decode score kernel (hy4_idx_score, kv_len from a per-step tensor) -> select_v2 (blockwise top-k, all-gather, merge) -> global top-2048 ids
        dense mode (every row <= 2048 keys): the host-built global ids [0 .. pos]
        ``id_filter``: global ids -> this column's local pool rows (+ replicate to the 8 owner blocks); shared layers reuse their index source's tensor (IndexCache)
    q all-gather over the columns -> Hy4 split kernel on the column's own pool keys (all 64 heads, emit stats) -> fused exchange + merge (HY4_FUSE=sparse,xchg: ONE XMerge per process, built
    in ``PoolCtx`` BEFORE the first trace capture) -> o_lat -> back (value up-projection, gate, o_proj, all-reduce).
All per-step data arrives in persistent device tensors (``PoolCtx.set_step``); the whole step is one trace per (mode, bucket).
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Hy4AttentionOpt
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

SCALE = 256**-0.5
NC = 8
K = 2048


def q_headrows(q, S):
    """stock-layout q [1,32,S,128] -> [S,1,32,128] TILE (tile row = head)."""
    x = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.permute(x, (2, 0, 1, 3))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _pad_rows(t, to):
    n = t.shape[2]
    return t if n == to else ttnn.pad(t, [(0, 0), (0, 0), (0, to - n), (0, 0)], 0)


_POOL_AG_LINKS = int(
    os.environ.get("HY4_POOL_AG_LINKS", "2")
)  # links of the pool path's all_gather_async (q gather, key gather); see slot-diag/DIAG.md


class PoolCtx:
    """Process-wide state of the pool decode path: persistent per-step device tensors, the shared XMerge, the split-kernel module, the select-pipeline parameters."""

    def __init__(
        self, md, mc, ccl, pages, rows=4, S=1, mode="topk", bucket_tokens=None, xchg=None, fuse=None, sel=None, xm=None
    ):
        self.md, self.mc, self.ccl, self.rows, self.S, self.pages = md, mc, ccl, rows, S, pages
        self.mode = mode
        self.bucket_tokens = bucket_tokens or pages * HP.CHUNK
        assert self.bucket_tokens % (8 * 32) == 0 and self.bucket_tokens <= pages * HP.CHUNK
        self.n_local = self.bucket_tokens // NC
        fz = os.environ.get("HY4_FUSE", "") if fuse is None else fuse
        self.fuse_sparse, self.xchg = "sparse" in fz, (("xchg" in fz) if xchg is None else xchg)
        self.fuse_idx = "idx" in fz
        self.sel = sel or dict(B=16, kb=512, kloc=512, split=4)
        up = lambda t, dt, shard_dim=1, row_dim=0: upload(
            md, t, dt, ttnn.ROW_MAJOR_LAYOUT, shard_dim=shard_dim, row_dim=row_dim
        )
        self.colid = up(HP.colid_host(rows), ttnn.uint32)
        self.colinfo = up(HP.colid_host(rows, [1 << 20] * rows), ttnn.uint32)
        self.wr = up(HP.write_rows([0] * rows, rows), ttnn.int32)
        self.upd = upload(
            md, torch.zeros(rows, NC, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, shard_dim=1, row_dim=0
        )
        self.kvcnt = up(HP.key_counts([0] * rows, rows), ttnn.uint32)
        self.ids_g = upload(
            md, torch.full((rows, 1, 1, K), -1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0
        )
        self.pt = upload(
            md, torch.arange(pages, dtype=torch.int32).reshape(1, pages), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.kcfg_key = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        from models.demos.blackhole.hy4_preview.tt import hy4_split

        self.mod = hy4_split
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        R = NC * S
        self.G, self.kc = hy4_split.pick_split(S)
        if "HY4_VR_G" in os.environ:  # probe override: split G / k_chunk
            self.G, self.kc = int(os.environ["HY4_VR_G"]), int(os.environ.get("HY4_VR_KC", self.kc))
        self.xm = xm  # an existing XMerge (tests that rebuild the ctx per bucket must share the ONE per-process object)
        if self.xchg and xm is None:
            from models.demos.blackhole.hy4_preview.tt import hy4_xmerge

            self.xm = hy4_xmerge.XMerge(
                md, S, nsplit=int(os.environ.get("HY4_XM_NSPLIT", "2"))
            )  # ONE per process, before the first trace capture
        self.ids8 = {}  # index-source layer -> idx8 tensor of the current step
        self.ids8_dense = None
        self.step_pos = None

    def set_bucket(self, tokens, mode=None):
        """decode bucket (call before capturing the trace of this bucket): the score kernel / select pipeline cover ``tokens`` tokens of the pool (tokens / 8 local rows per column)."""
        assert tokens % 256 == 0 and tokens <= self.pages * HP.CHUNK
        self.bucket_tokens, self.n_local = tokens, tokens // NC
        if mode is not None:
            self.mode = mode

    # ---- host side (outside traces)
    def set_step(self, pos):
        """pos [rows]: position of the token being decoded per user (keys 0..pos valid after the append)."""
        md = self.md
        up = lambda t, dt, tgt, shard_dim=1, row_dim=0: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                t,
                dtype=dt,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(row_dim, shard_dim), mesh_shape=tuple(md.shape)),
            ),
            tgt,
        )
        pos = [int(p) for p in pos]
        wr = HP.write_rows(pos, self.rows)
        up(wr, ttnn.int32, self.wr)
        up(wr[:, :, 0, 0], ttnn.int32, self.upd)  # [rows, 8]
        up(HP.key_counts(pos, self.rows), ttnn.uint32, self.kvcnt)
        up(HP.colid_host(self.rows, [p + 1 for p in pos]), ttnn.uint32, self.colinfo)
        ids = torch.full((self.rows, 1, 1, K), -1, dtype=torch.int32)
        for r, p in enumerate(pos):
            n = min(p + 1, K)
            ids[r, 0, 0, :n] = torch.arange(n, dtype=torch.int32)
        up(ids, ttnn.uint32, self.ids_g, shard_dim=None, row_dim=0)
        self.step_pos = pos

    # ---- device side (traceable)
    def ag(self, t):
        return ttnn.experimental.all_gather_async(
            t,
            dim=3,
            cluster_axis=1,
            mesh_device=self.md,
            topology=ttnn.Topology.Ring,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            num_links=_POOL_AG_LINKS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    def gather_q(self, q_rm):
        return ttnn.experimental.all_gather_async(
            q_rm,
            dim=2,
            cluster_axis=1,
            mesh_device=self.md,
            topology=ttnn.Topology.Ring,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            num_links=_POOL_AG_LINKS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    def dense_ids8(self):
        """host-built global ids -> local per-column ids of every user (dense mode: shared by all layers of the step)."""
        if self.ids8_dense is None:  # persistent output (no allocation inside traces)
            self.ids8_dense = self.new_idx8()
        HP.id_filter(self.ids_g, self.colinfo, self.S, out=self.ids8_dense)
        return self.ids8_dense

    def new_idx8(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, NC * self.S, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.md, ttnn.DRAM_MEMORY_CONFIG
        )

    def attend(self, qfull, kv_pool, idx8, sink):
        """qfull [1,8,S,576] TILE bf16 (the column's 8 heads) -> o_lat [1,8,S,512] TILE bf16 (merged over the 8 columns, sink included)."""
        S = self.S
        q_rm = ttnn.to_layout(qfull, ttnn.ROW_MAJOR_LAYOUT)
        qg = self.gather_q(q_rm)  # [1,8,8S,576]
        ttnn.deallocate(q_rm)
        out = self.mod.sparse_sdpa_split(
            qg, kv_pool, idx8, 512, SCALE, self.kc, self.G, self.ckc, None, emit_stats=3, owner_s=S
        )
        ttnn.deallocate(qg)
        if self.xm is not None:
            res = self.xm(out, sink)
        elif self.fuse_sparse:
            from models.demos.blackhole.hy4_preview.tt import hy4_merge

            x = ttnn.experimental.all_to_all_async_generic(
                out, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            res = hy4_merge.merge_fused(x, sink, S)
        else:
            x = ttnn.experimental.all_to_all_async_generic(
                out, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            res = merge3(x, sink, S)
        # res [1,S,32,512] TILE: rows 0..7 = the 8 real heads of user s -> [1,8,S,512]
        r = ttnn.to_layout(res, ttnn.ROW_MAJOR_LAYOUT)
        r = ttnn.slice(r, [0, 0, 0, 0], [1, S, 8, 512])  # [1,S,8,512]
        r = ttnn.permute(r, (0, 2, 1, 3))  # [1,8,S,512]
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT)


def select_global(ctx, qh, w32, key_view):
    """decode indexer chain on the key pool: score kernel (kv_len per column from ctx.kvcnt) -> select_v2 -> global top-2048 ids uint32 [S,1,1,2048] (identical on all columns).
    qh [S,1,32,128] TILE (heads = tile rows), w32 [1,1,32,32] TILE (row = query row, col = head), key_view [1,1,pages*64,128] bfp8 TILE.
    """
    from models.demos.blackhole.hy4_preview.tt import hy4_idx_score as IS
    from models.demos.blackhole.hy4_preview.tt.hy4_idx_select import select_v2

    S, n = ctx.S, ctx.n_local
    score = IS.idx_score(qh, key_view, w32, 0, S, kvt=ctx.kvcnt, n_rows=n)  # [1,1,S,n]
    sel = ctx.sel
    B = max(1, min(sel["B"], n // sel["kb"]))
    kb, kloc = min(sel["kb"], n // B), min(sel["kloc"], n)
    gids = select_v2(ctx.ag, ctx.colid, score, n, B, kb, kloc, K, sel["split"], False, None)
    ttnn.deallocate(score)
    return gids


def merge3(x, sinkcol, S):
    """stock-op merge (no fused kernel): x [1,8*S,32,576] TILE (src-major then user), sinkcol [1,S,32,1] -> [1,S,32,512]"""
    V = 512
    xr = ttnn.reshape(x, [NC, S, 32, V + 64])
    m = ttnn.slice(xr, [0, 0, 0, V], [NC, S, 32, V + 1])
    l = ttnn.slice(xr, [0, 0, 0, V + 32], [NC, S, 32, V + 33])
    w = ttnn.multiply(l, ttnn.exp(ttnn.subtract(m, sinkcol)))
    den = ttnn.add(ttnn.sum(w, dim=0, keepdim=True), 1.0)
    f = ttnn.multiply(w, ttnn.reciprocal(den))
    acc = ttnn.sum(ttnn.multiply(xr, f), dim=0, keepdim=True)
    return ttnn.slice(acc, [0, 0, 0, 0], [1, S, 32, V])


class Hy4AttentionPool(Hy4AttentionOpt):
    """Hy4AttentionOpt + the pool decode step. Prefill and the replicated-cache decode paths are unchanged."""

    def init_pool(self, ctx, kv_pool, key_pool4=None):
        self.ctx, self.kv_pool, self.key_pool4 = ctx, kv_pool, key_pool4
        S = ctx.S
        self.idx8_buf = (
            ctx.new_idx8() if self.has_indexer else None
        )  # this layer's local ids (persistent; shared layers of this source read it)
        self.key_view = ttnn.reshape(key_pool4, [1, 1, ctx.pages * 64, 128]) if key_pool4 is not None else None
        so = self.host_w["sink_over_scale"].tensor.reshape(-1).float()  # [64] sink / scale per head
        t = torch.zeros(1, S, 32, NC)
        for c in range(NC):
            t[0, :, :8, c] = (so[8 * c : 8 * c + 8] * SCALE)[None, :]
        self.sink_pool = upload(self.md, t.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3)

    # ------------------------------------------------------------------ pieces
    def pool_append_kv(self, lat, k_pe):
        dp = ttnn.experimental.deepseek_prefill
        dram = (
            ttnn.DRAM_MEMORY_CONFIG
        )  # the stock quantiser / packer want DRAM-interleaved inputs (HY4_OPT=l1 keeps activations in L1)
        lat = ttnn.to_memory_config(lat, dram)
        f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
        rope_rm = ttnn.to_layout(ttnn.to_memory_config(k_pe, dram), ttnn.ROW_MAJOR_LAYOUT)
        packed = dp.pack_scaled_fp8_kv_cache(f8, sc, rope_rm)
        HP.row_write(packed, self.ctx.wr, self.kv_pool, n=int(packed.shape[2]))
        for t in (f8, sc, rope_rm, packed):
            ttnn.deallocate(t)

    def pool_append_key(self, ik, o, tabs):
        ix, ctx = self.indexer, self.ctx
        cos, sin, icos, isin, c128, s128 = tabs
        k = ttnn.layer_norm(ik, weight=ix.k_norm_w, bias=ix.k_norm_b, epsilon=ix.ln_eps)
        k = self._idx_rot(o, k, icos, isin, c128, s128)
        kd = ttnn.to_memory_config(k, ctx.kcfg_key)
        ttnn.experimental.paged_update_cache(self.key_pool4, kd, update_idxs_tensor=ctx.upd, page_table=ctx.pt)
        ttnn.deallocate(kd)
        ttnn.deallocate(k)

    def pool_select(self, d, o, tabs):
        """global top-2048 ids [S,1,1,2048] (natural positions) of this full-indexer layer from the key pool."""
        ctx, S = self.ctx, self.ctx.S
        q = self.idx_q(o, d, tabs)  # [1,32,S,128]
        qh = q_headrows(q, S)
        ttnn.deallocate(q)
        w32 = _pad_rows(d["iw"], 32)
        gids = select_global(ctx, qh, w32, self.key_view)
        for t in (qh, w32):
            ttnn.deallocate(t)
        return gids

    # ------------------------------------------------------------------ step
    def decode_step_pool(self, h, o, tabs, dense_ids8=None):
        """h [1,1,S,6144] (post input norm). Returns out [1,1,S,6144]; the layer's idx8 is left in ctx.ids8[layer]."""
        ctx = self.ctx
        d = self.front(h, o, tabs)
        self.pool_append_kv(d["lat"], d["k_pe"])
        for t in (d["lat"], d["k_pe"]):
            ttnn.deallocate(t)
        if self.has_indexer:
            self.pool_append_key(d["ik"], o, tabs)
            if ctx.mode == "dense":
                idx8 = dense_ids8
            else:
                gids = self.pool_select(d, o, tabs)
                idx8 = HP.id_filter(gids, ctx.colinfo, ctx.S, out=self.idx8_buf)
                ttnn.deallocate(gids)
            ctx.ids8[self.layer] = idx8
        else:
            idx8 = dense_ids8 if ctx.mode == "dense" else ctx.ids8[self.spec.indexer_source]
        o_lat = ctx.attend(d["qfull"], self.kv_pool, idx8, self.sink_pool)
        ttnn.deallocate(d["qfull"])
        out = self.back(o, o_lat, d["g"])
        ttnn.deallocate(o_lat)
        return out
