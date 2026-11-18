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
    if os.environ.get("HY4_IDX_QFAST", "0") == "1":  # one TILE permute instead of to_layout + permute + to_layout
        return ttnn.permute(q, (2, 0, 1, 3))
    x = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.permute(x, (2, 0, 1, 3))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _pad_rows(t, to):
    n = t.shape[2]
    return t if n == to else ttnn.pad(t, [(0, 0), (0, 0), (0, to - n), (0, 0)], 0)


_POOL_AG_LINKS = int(
    os.environ.get("HY4_POOL_AG_LINKS", "2")
)  # links of the pool path's all_gather_async (q gather, key gather); see slot-diag/DIAG.md


_XMC = {}  # B26: one compact exchange object per (mesh, S)
_QP = {}  # B29: one q push object per (mesh, S)
_XPU = {}  # B47a: one XPUSH object per (mesh, S)


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
        self.colinfo = up(HP.colid_host(rows, [1 << 20] * rows, S), ttnn.uint32)
        self.TR = (S + 3) // 4
        self.perrow = S > 1  # verify rows: per-row causal limits in the score kernel (kvcnt) and in id_filter (colinfo)
        self.wr = up(HP.write_rows([0] * rows, rows, S), ttnn.int32)
        # verify rows (S = k+1 query rows per user, positions pos..pos+S-1): one key-append index tensor per row (the S appends run one after the other: two rows of the same key tile must not race)
        self.upd_rows = [
            upload(
                md, torch.zeros(rows, NC, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, shard_dim=1, row_dim=0
            )
            for _ in range(S)
        ]
        self.upd = self.upd_rows[0]
        self.kvcnt = up(
            HP.key_counts([0] * rows, rows, S), ttnn.uint32
        )  # element j = valid local keys of verify row j (causal)
        self.ids_g = upload(
            md, torch.full((rows * S, 1, 1, K), -1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0
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
        if self.xchg and xm is None and S > int(os.environ.get("HY4_XM_MAXS", "1")):
            # verify rows (S > 1): ONE XMerge per process (the S=1 decode context owns it) and the exchange is slower than the compact a2a + A1 merge at S >= 2 (verify-rows lat: +12..+60 us): A1 merge here
            self.xchg = False
        if self.xchg and xm is None:
            from models.demos.blackhole.hy4_preview.tt import hy4_xmerge

            self.xm = hy4_xmerge.XMerge(
                md, S, nsplit=int(os.environ.get("HY4_XM_NSPLIT", "2"))
            )  # ONE per process, before the first trace capture
        # compact attend (HY4_VR_COMPACT: 1 on, 0 off, default on for S >= 2): the split kernel packs the 8 real head rows of up to 4 users into one 32-row tile (all_to_all payload 4x smaller), the A1 merge works on TR = ceil(S/4) tile rows
        self.compact = (os.environ.get("HY4_VR_COMPACT", "1") == "1") and S > 1 and not self.xchg
        self.hm_compact = (
            self.compact and os.environ.get("HY4_ATTN_HM", "0") == "1" and S <= 16
        )  # HY4_ATTN_HM=1: head-major merge output (no post-merge layout glue)
        # B26 (default OFF): HY4_B26_XCOMPACT=1: the compact verify layout goes through the fabric exchange + merge program (XMerge users=S) instead of all_to_all + A1 merge
        self.xmc = None
        if self.compact and self.hm_compact and int(os.environ.get("HY4_B26_XCOMPACT", "0")) == 1:
            key = (id(md), S)
            if key not in _XMC:
                from models.demos.blackhole.hy4_preview.tt import hy4_xmerge

                _XMC[key] = hy4_xmerge.XMerge(md, self.TR, nsplit=int(os.environ.get("HY4_XM_NSPLIT", "8")), users=S)
            self.xmc = _XMC[key]
        # B29 (default OFF): HY4_B29_QPUSH=1: the q all-gather is a push-only program + a flag wait in the split reader (hy4_qpush); owner layout (S_user rows) only
        self.qpush = None
        if int(os.environ.get("HY4_B29_QPUSH", "0")) == 1 and self.compact:
            key = (id(md), S)
            if key not in _QP:
                from models.demos.blackhole.hy4_preview.tt import hy4_qpush

                _QP[key] = hy4_qpush.QPush(md, S)
            self.qpush = _QP[key]
        # B47a (default OFF): HY4_B47A_XPUSH=1: the cross-column exchange runs inside the split program (comm cores + flag-waiting merge reader), compact verify layout with head-major merge output, S <= 4
        self.xpu = None
        if int(os.environ.get("HY4_B47A_XPUSH", "0")) == 1 and self.compact and self.hm_compact and S <= 4:
            key = (id(md), S)
            if key not in _XPU:
                from models.demos.blackhole.hy4_preview.tt import hy4_xpush

                _XPU[key] = hy4_xpush.XPush(md, S)
            self.xpu = _XPU[key]
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
        S = self.S
        wr = HP.write_rows(pos, self.rows, S)
        up(wr, ttnn.int32, self.wr)
        for j in range(S):
            up(wr[:, :, 0, j], ttnn.int32, self.upd_rows[j])  # [rows, 8]
        up(HP.key_counts(pos, self.rows, S), ttnn.uint32, self.kvcnt)
        up(HP.colid_host(self.rows, [p + 1 for p in pos], S), ttnn.uint32, self.colinfo)
        ids = torch.full((self.rows * S, 1, 1, K), -1, dtype=torch.int32)
        for r, p in enumerate(pos):
            for j in range(S):  # verify row j sees the tokens 0..p+j
                n = min(p + j + 1, K)
                ids[r * S + j, 0, 0, :n] = torch.arange(n, dtype=torch.int32)
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

    def id_push(self, kloc):
        """B35: Push object of the local candidate id rows [S,1,1,kloc] u32 (one per kloc, cached)"""
        from models.demos.blackhole.hy4_preview.tt import hy4_qpush

        c = self.__dict__.setdefault("_idp", {})
        if kloc not in c:
            pb = kloc * 4
            c[kloc] = hy4_qpush.Push(self.md, self.S, pb, max(1, min(4352 // pb, 2)))
        return c[kloc]

    def val_push(self, kloc):
        """B35: Push object of the local candidate score rows [S,1,1,kloc] bf16 (cached per kloc)"""
        from models.demos.blackhole.hy4_preview.tt import hy4_qpush

        c = self.__dict__.setdefault("_vp", {})
        if kloc not in c:
            pb = kloc * 2
            c[kloc] = hy4_qpush.Push(self.md, self.S, pb, max(1, min(4352 // pb, 4)))
        return c[kloc]

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
        HP.id_filter(self.ids_g, self.colinfo, self.S, out=self.ids8_dense, perrow=self.perrow)
        return self.ids8_dense

    def make_sink(self, sink8):
        """sink8 [64] = sink logit (scaled domain) per head -> the sink tensor of ``attend``: per-user layout [1,S,32,8] (shard dim 3: column c holds its 8 heads in rows 0..7 of every user tile), or the compact layout [1,TR,32,8]
        (user s = 4u+b: rows 8b..8b+7 of tile row u)."""
        n = self.TR if self.compact else self.S
        t = torch.zeros(1, n, 32, NC)
        for c in range(NC):
            if self.compact:
                for b in range(4):
                    t[0, :, 8 * b : 8 * b + 8, c] = sink8[8 * c : 8 * c + 8][None, :]
            else:
                t[0, :, :8, c] = sink8[8 * c : 8 * c + 8][None, :]
        return upload(self.md, t.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3)

    def new_idx8(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, NC * self.S, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.md, ttnn.DRAM_MEMORY_CONFIG
        )

    def attend(self, qfull, kv_pool, idx8, sink, q_rm=False, q_pushed=False):
        """qfull [1,8,S,576] TILE bf16 (the column's 8 heads; q_rm=True: already ROW_MAJOR, B17 fused q-prep) -> o_lat [1,8,S,512] TILE bf16 (merged over the 8 columns, sink included)."""
        S = self.S
        q_rm = qfull if q_rm else ttnn.to_layout(qfull, ttnn.ROW_MAJOR_LAYOUT)
        if self.qpush is not None and q_pushed:
            qg = self.qpush.land  # B47b: the push already ran inside the fused q-prep program
        elif self.qpush is not None:
            qg = self.qpush(q_rm)  # persistent landing tensor (flag-gated read in the split reader)
        else:
            qg = self.gather_q(q_rm)  # [1,8,8S,576]
        ttnn.deallocate(q_rm)
        if self.xpu is not None:
            from models.demos.blackhole.hy4_preview.tt import hy4_merge

            out = self.xpu.split(
                qg,
                kv_pool,
                idx8,
                self.mod,
                self.kc,
                self.G,
                self.ckc,
                SCALE,
                qland=(self.qpush, S) if self.qpush is not None else None,
            )
            if self.qpush is None:
                ttnn.deallocate(qg)
            res = hy4_merge.merge_fused(self.xpu.land, sink, self.TR, full=S > 2, hm_users=S, xp=self.xpu)
            ttnn.deallocate(out)
            return res
        out = self.mod.sparse_sdpa_split(
            qg,
            kv_pool,
            idx8,
            512,
            SCALE,
            self.kc,
            self.G,
            self.ckc,
            None,
            emit_stats=3,
            owner_s=S,
            compact=self.compact and self.xm is None,
            **({"qland": (self.qpush, S)} if self.qpush is not None else {}),
        )
        if self.qpush is None:
            ttnn.deallocate(qg)
        if self.compact and self.xm is None:
            if self.xmc is not None:
                return self.xmc(out, sink)  # head-major [1,8,S,512] TILE (persistent) from the exchange + merge program
            from models.demos.blackhole.hy4_preview.tt import hy4_merge

            TR = self.TR
            x = ttnn.experimental.all_to_all_async_generic(
                out, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            if self.hm_compact:
                return hy4_merge.merge_fused(
                    x, sink, TR, full=S > 2, hm_users=S
                )  # head-major [1,8,S,512] TILE (persistent) straight from the merge writer
            res = hy4_merge.merge_fused(x, sink, TR, full=S > 2)  # [1,TR,32,512]: rows 8*(s%4)+h of tile row s//4
            r = ttnn.to_layout(res, ttnn.ROW_MAJOR_LAYOUT)
            r = ttnn.reshape(r, [TR * 4, 8, 512])
            if TR * 4 != S:
                r = ttnn.slice(r, [0, 0, 0], [S, 8, 512])
            r = ttnn.permute(r, (1, 0, 2))  # [8,S,512]
            r = ttnn.reshape(r, [1, 8, S, 512])
            return ttnn.to_layout(r, ttnn.TILE_LAYOUT)
        if self.xm is not None:
            res = self.xm(out, sink)
            if getattr(self.xm, "hm", False):
                return res  # head-major [1,8,S,512] TILE straight from the merge writer (persistent: the caller must not deallocate it)
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


def append_key_rows(ctx, key_pool4, k):
    """k [1,1,S,128] TILE (S verify rows of the step, already norm + rope) -> the key pool at the S positions of ``ctx.upd_rows``: one ``paged_update_cache`` per row, in order (two rows of the same bfp8 tile must not race)."""
    if ctx.S > 1 and ctx.S <= 4:
        from models.demos.blackhole.hy4_preview.tt import hy4_key_append as KA

        if KA.enabled():  # B32: the S appends as ONE one-core program (bit-identical)
            kk = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG) if k.memory_config().is_sharded() else k
            KA.key_append(kk, key_pool4, ctx.upd_rows, ctx.pt, ctx.S)
            if kk is not k:
                ttnn.deallocate(kk)
            return
    for j in range(ctx.S):
        kj = k if ctx.S == 1 else ttnn.slice(k, [0, 0, j, 0], [1, 1, j + 1, 128])
        kd = ttnn.to_memory_config(kj, ctx.kcfg_key)
        ttnn.experimental.paged_update_cache(key_pool4, kd, update_idxs_tensor=ctx.upd_rows[j], page_table=ctx.pt)
        ttnn.deallocate(kd)
        if kj is not k:
            ttnn.deallocate(kj)


def select_global(ctx, qh, w32, key_view, fin_out=None):
    """decode indexer chain on the key pool: score kernel (kv_len per column from ctx.kvcnt) -> select_v2 -> global top-2048 ids uint32 [S,1,1,2048] (identical on all columns).
    qh [S,1,32,128] TILE (heads = tile rows), w32 [1,1,32,32] TILE (row = query row, col = head), key_view [1,1,pages*64,128] bfp8 TILE.
    """
    from models.demos.blackhole.hy4_preview.tt import hy4_idx_score as IS
    from models.demos.blackhole.hy4_preview.tt.hy4_idx_select import select_v2

    S, n = ctx.S, ctx.n_local
    sel = ctx.sel
    B = max(1, min(sel["B"], n // sel["kb"]))
    kb, kloc = min(sel["kb"], n // B), min(sel["kloc"], n)
    # B37: HY4_IDX_BLK=1 at block width 8192 (1M pool, B=16) is broken in the model (RESULT_B37). HY4_B37_BLKMODE = both (default, as before) | score (blocked score write only) | cand (blocked cs_cand2 output only) | off
    blk_on = os.environ.get("HY4_IDX_BLK", "0") == "1" and B > 1
    bm = os.environ.get("HY4_B37_BLKMODE", "both")
    if (n // B) > int(os.environ.get("HY4_IDX_BLK_MAXM", "1000000")):
        bm = "off"
    if (
        blk_on and n % (32 * B) != 0
    ):  # B37 ROOT CAUSE guard: a block boundary inside a 32-key tile strip makes the score writer issue misaligned page-split writes (garbage scores)
        if not getattr(select_global, "_warned", False):
            select_global._warned = True
            print(
                f"HY4 WARNING: blocked indexer layout disabled: n_local={n} is not a multiple of 32*B={32 * B} (use pool pages % 8 == 0)",
                flush=True,
            )
        bm = "off"
    blk = blk_on and bm in ("both", "score")
    blk_out = blk_on and bm in ("both", "cand")
    score = IS.idx_score(
        qh, key_view, w32, 0, S, kvt=ctx.kvcnt, n_rows=n, kvt_rows=ctx.perrow, blocks=B if blk else 1
    )  # [1,1,S,n]
    fin = None
    _nd = int(os.environ.get("HY4_B38_DUMMY_SEM", "0"))
    if (
        _nd > 0 and "_dsem" not in ctx.__dict__
    ):  # B38 control experiment: same persistent L1 footprint as a Push (global semaphores at the same point in time), no protocol
        _g = ctx.md.compute_with_storage_grid_size()
        _cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_g.x - 1, _g.y - 1))])
        _kind = os.environ.get(
            "HY4_B39_DUMMY_KIND", "sem_l1"
        )  # B39: sem_l1 (B38 control) | sem_small (same semaphores in L1_SMALL) | tensor (a plain persistent L1 tensor of 64 B/bank per dummy semaphore)
        if _kind == "sem_small":
            ctx._dsem = [ttnn.create_global_semaphore(ctx.md, _cr, 0, ttnn.BufferType.L1_SMALL) for _ in range(_nd)]
        elif _kind == "tensor":
            _nb = _g.x * _g.y
            ctx._dsem = [
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, _nb, 32 * _nd]),
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                    ctx.md,
                    ttnn.L1_MEMORY_CONFIG,
                )
            ]
        else:
            ctx._dsem = [ttnn.create_global_semaphore(ctx.md, _cr, 0) for _ in range(_nd)]
    if (
        fin_out is not None
    ):  # B35: gather + id_filter fused (HY4_B35_GFILT=1), optionally with the id all-gather replaced by a push (HY4_B35_IDPUSH=1)
        own = (
            os.environ.get("HY4_B35_OWN", "0") == "1"
        )  # no id all-gather at all (the filter keeps only this column's own block)
        fin = dict(
            colinfo=ctx.colinfo,
            out=fin_out,
            perrow=ctx.perrow,
            own=own,
            sel3=os.environ.get("HY4_B36_SEL3", "0") == "1",
            vpush=ctx.val_push(kloc) if os.environ.get("HY4_B35_VPUSH", "0") == "1" else None,
            push=ctx.id_push(kloc) if (os.environ.get("HY4_B35_IDPUSH", "0") == "1" and not own) else None,
        )
    gids = select_v2(
        ctx.ag,
        ctx.colid,
        score,
        n,
        B,
        kb,
        kloc,
        K,
        sel["split"],
        False,
        None,
        blocked=blk,
        fin=fin,
        blocked_out=blk_out,
    )
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
        self.sink_pool = ctx.make_sink(so * SCALE)

    # ------------------------------------------------------------------ pieces
    def pool_append_kv(self, lat, k_pe):
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA

        S = int(lat.shape[2])
        if (
            FA.kv_enabled()
            and S <= 8
            and not lat.memory_config().is_sharded()
            and not k_pe.memory_config().is_sharded()
            and lat.layout == ttnn.TILE_LAYOUT
            and k_pe.layout == ttnn.TILE_LAYOUT
            and lat.dtype == ttnn.bfloat16
            and k_pe.dtype == ttnn.bfloat16
        ):  # HY4_KV_FUSE=1: quantise + pack + row write in ONE program
            FA.fused_kv_append(lat, k_pe, self.ctx.wr, self.kv_pool, S)
            return
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
        append_key_rows(ctx, self.key_pool4, k)
        ttnn.deallocate(k)

    def pool_select(self, d, o, tabs):
        """global top-2048 ids [S,1,1,2048] (natural positions) of this full-indexer layer from the key pool."""
        ctx, S = self.ctx, self.ctx.S
        q = self.idx_q(o, d, tabs)  # [1,32,S,128]
        qh = q_headrows(q, S)
        ttnn.deallocate(q)
        w32 = (
            d["iw"] if os.environ.get("HY4_IDX_QFAST", "0") == "1" else _pad_rows(d["iw"], 32)
        )  # QFAST: the score kernel reads tile pages and only rows r < S; the tile padding rows are never used
        fin_out = (
            self.idx8_buf
            if (
                (os.environ.get("HY4_B35_GFILT", "0") == "1" or os.environ.get("HY4_B35_OWN", "0") == "1")
                and 1 < S <= 4
                and ctx.mode != "dense"
            )
            else None
        )
        gids = select_global(ctx, qh, w32, self.key_view, fin_out)
        for t in (qh, w32):
            ttnn.deallocate(t)
        return gids

    # ------------------------------------------------------------------ step
    def decode_step_pool(self, h, o, tabs, dense_ids8=None):
        """h [1,1,S,6144] (post input norm). Returns out [1,1,S,6144]; the layer's idx8 is left in ctx.ids8[layer]."""
        ctx = self.ctx
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA

        qrm = (
            FA.enabled("q")
            and ctx.S <= 4
            and o.rope1
            and self.w_uk.dtype == ttnn.bfloat16
            and tuple(self.w_uk.shape)[1:] == (8, 192, 512)
        )
        qfold = (
            qrm
            and ctx.qpush is not None
            and os.environ.get("HY4_B47B_QFOLD", "0") == "1"
            and FA.kv_in_q()
            and ctx.S <= 4
        )  # B47b: the q push runs inside the fused q-prep program
        d = self.front(h, o, tabs, qrm=qrm, kvf=(ctx.wr, self.kv_pool), qpush=ctx.qpush if qfold else None)
        if not d.get("kv_done"):
            self.pool_append_kv(d["lat"], d["k_pe"])
        for t in (d["lat"], d["k_pe"]):
            ttnn.deallocate(t)
        if self.has_indexer:
            self.pool_append_key(d["ik"], o, tabs)
            if ctx.mode == "dense":
                idx8 = dense_ids8
            else:
                gids = self.pool_select(d, o, tabs)
                if gids is self.idx8_buf:  # B35: already filtered (fused)
                    idx8 = gids
                else:
                    idx8 = HP.id_filter(gids, ctx.colinfo, ctx.S, out=self.idx8_buf, perrow=ctx.perrow)
                    ttnn.deallocate(gids)
            ctx.ids8[self.layer] = idx8
        else:
            idx8 = dense_ids8 if ctx.mode == "dense" else ctx.ids8[self.spec.indexer_source]
        o_lat = ctx.attend(
            d["qfull"], self.kv_pool, idx8, self.sink_pool, q_rm=d.get("qrm", False), q_pushed=d.get("qpushed", False)
        )
        ttnn.deallocate(d["qfull"])
        out = self.back(o, o_lat, d["g"], d.get("gfold", False))
        if not (getattr(ctx.xm, "hm", False) or ctx.hm_compact):
            ttnn.deallocate(o_lat)
        return out
