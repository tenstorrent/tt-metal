# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Column-sharded decode indexer key slab (``DSV41_IDX_KEY_SHARD=1``, default off; wired into ``DSV41DecodeIndexer`` / ``SpecIndexer``, ``dsv41_model.Model._build_indexer``).

The key slab of an index-source layer is [U,1,n_alloc,128] per mesh row, replicated over the ``cols`` mesh columns. Here the slab is sharded over the columns by an ENTRY INTERLEAVE
(entry e lives on column e % cols at local slot e // cols: the layout of the column-sharded prefill key FIFO, ``prefill_sparse.PrefillIndexer.keys_full``), so that

  * "the first nb entries" (decode position buckets) are the first nb / cols local slots on EVERY column (balanced work at every context length),
  * the valid entries of every column are a prefix (valid count of column c: ceil((N - c) / cols)): the fused score kernel (tt/idx_score.py) reads that count on the device,
  * global id = cols * local_id + c  (exact in float32 up to 2**24),
  * the prefill key FIFO has the same interleave, so the hand-off is column-local when the FIFO offset is a multiple of cols (``dsv41_model._export_user_keys_cs``).

Selection: each column scores its n_alloc / cols keys (the same matmul / fused score code as the replicated path on the local slab, per-row valid LOCAL counts), takes a local top-k, the (score, global
id) candidates of the cols columns are all-gathered over the mesh columns (R x cols*k) and a global top-k is taken among them. The global top-k is contained in the union of the local top-k, so the
selected SET equals the replicated one except for the choice among entries TIED at the k-th score (bf16 scores tie at the cutoff in most rows at long context; which of the tied entries the replicated
``topk_large_indices`` returns is a function of its chunk / segment merge tree, not of the entry ids, and cannot be reproduced from candidate lists: tests/test_idx_cs_merge.py ``test_tie_probe``). The KV pool
stays replicated, so the ids index it directly. The merge (``cs_merge_fast``, default; DSV41_IDX_CS_FAST=0 selects the op chain of ``cs_select``) is two JIT data-movement kernels (tt/idx_cs_kernels: candidate
values + global ids, final gather + ascending id sort) around two all_gathers and the merge top-k.
"""

import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idx_cs_kernels")


def _cs_fast():
    """DSV41_IDX_CS_FAST (default 1): the candidate stage / final stage of the merge as two JIT data-movement kernels (tt/idx_cs_kernels) instead of ~40 small ops."""
    return os.environ.get("DSV41_IDX_CS_FAST", "1") == "1"


def _cs_prog(name, mesh, rows, cbs_spec, ct, accs, common, tensors, tag):
    """one-kernel generic_op program, one core per row (rows <= cores); cbs_spec: [(pages, page_bytes, dtype)] in CB index order."""
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][: min(rows, grid.x * grid.y)]
    nc = len(cores)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for t, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [t, nc, len(range(t, rows, nc))]
    cbs = [_cb(core_set, i, pg, pb, dt) for i, (pg, pb, dt) in enumerate(cbs_spec)]
    acc = []
    for t in accs:
        acc += _acc(t)
    kern = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=list(range(len(cbs_spec))) + ct + acc,
        runtime_args=rt,
        common_runtime_args=common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(tag, rows, tuple(ct), tuple(tuple(_acc(t)) for t in accs))
    ttnn.generic_op(tensors, prog)


def cs_cand(sc, lids, colid, cols):
    """sc bf16 RM [R,1,1,n] (the local scores of this column), lids uint32 RM [R,1,1,K] (local top-K slots) -> (vals bf16 RM [R,1,1,K], gids uint32 RM [R,1,1,K] = slot * cols + column);
    colid uint32 RM [1,1,1,16] per device (word 0 = the column index)."""
    R, n, K = int(sc.shape[0]), int(sc.shape[-1]), int(lids.shape[-1])
    mesh = sc.device()
    assert (
        sc.dtype == ttnn.bfloat16 and sc.layout == ttnn.ROW_MAJOR_LAYOUT and lids.dtype == ttnn.uint32 and n % 32 == 0
    )
    if os.environ.get("DSV41_OL_CSCAND", "0") == "1" and K % 128 == 0 and n * 2 <= 98304 and R * 4 <= 110:
        from models.demos.blackhole.deepseek_v41_flash.tt import ol_idx

        return ol_idx.cs_cand_fast(sc, lids, colid, cols, split=4)
    vals = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    gids = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    u32, bf = ttnn.uint32, ttnn.bfloat16
    _cs_prog(
        "cs_cand.cpp",
        mesh,
        R,
        [(1, K * 4, u32), (1, K * 64, u32), (1, K * 2, bf), (1, K * 4, u32), (1, 64, u32)],
        [K, cols, n],
        [sc, lids, colid, vals, gids],
        [
            sc.buffer_address(),
            lids.buffer_address(),
            colid.buffer_address(),
            vals.buffer_address(),
            gids.buffer_address(),
        ],
        [sc, lids, colid, vals, gids],
        0xC51,
    )
    return vals, gids


def cs_final(pos, cand, max_id=0):
    """pos uint32 RM [R,1,1,K] (merge top-K positions into the W gathered candidates), cand uint32 RM [R,1,1,W] (their global ids) -> ids uint32 RM [R,1,1,K], ascending (sentinel last)."""
    R, K, W = int(pos.shape[0]), int(pos.shape[-1]), int(cand.shape[-1])
    mesh = pos.device()
    assert pos.dtype == ttnn.uint32 and cand.dtype == ttnn.uint32 and K % 2 == 0
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    u32 = ttnn.uint32
    nw = (max_id + 32) // 32 if (max_id and os.environ.get("DSV41_IDX_CS_SORT", "bitmap") == "bitmap") else 0
    cnt_bytes = max(256 * 4, (nw + (nw + 31) // 32) * 4)
    _cs_prog(
        "cs_final.cpp",
        mesh,
        R,
        [(1, K * 4, u32), (1, W * 4, u32), (1, K * 4, u32), (1, K * 4, u32), (1, cnt_bytes, u32)],
        [K, W, nw],
        [pos, cand, out],
        [pos.buffer_address(), cand.buffer_address(), out.buffer_address()],
        [pos, cand, out],
        0xC52,
    )
    return out


def cs_merge_fast(ix, consts, sr, R, n, vl):
    """local top-k of the row-major scores ``sr`` [R,1,1,n] (valid local slots ``vl``), candidate stage kernel, all_gather over the mesh columns, merge top-k, final stage kernel
    -> ids uint32 RM [R,1,1,k] GLOBAL entry ids, ASCENDING (the canonical order: ``indexer.canonical_ids`` is not needed).
    """
    k, cols = ix.topk, ix.cols
    lids = ttnn.experimental.topk_large_indices(ttnn.reshape(sr, [1, 1, R, n]), k=k, valid_length_tensor=vl)
    lids = ttnn.reshape(lids if lids.dtype == ttnn.uint32 else ttnn.typecast(lids, ttnn.uint32), [R, 1, 1, k])
    vals, gids = cs_cand(sr, lids, consts["col_u32"], cols)
    cv = ix.mc.allgather(vals, ix.ccl, axis=1, dim=3)  # [R,1,1,cols*k]
    ci = ix.mc.allgather(gids, ix.ccl, axis=1, dim=3)
    W = int(cv.shape[-1])
    pos = ttnn.experimental.topk_large_indices(ttnn.reshape(cv, [1, 1, R, W]), k=k)
    pos = ttnn.reshape(pos if pos.dtype == ttnn.uint32 else ttnn.typecast(pos, ttnn.uint32), [R, 1, 1, k])
    return cs_final(pos, ci, int(ix.n_alloc) * cols)


def cs_select(ix, consts, scores, R, n, nv, vl, want_vals=False):
    """Selection over a column-sharded key slab, shared by ``ColShardSelect`` (the unit test) and the model's indexers (``DSV41DecodeIndexer._cs_select``).

    ``ix``: owner of ``topk`` / ``cols`` / ``mc`` (mesh config) / ``ccl`` / ``md`` / ``n_alloc`` (local slab length); ``consts``: ``_cs_consts`` of the device; ``scores`` [R,1,1,n] row-major bf16 (the local scores of
    the first ``n`` local slots); ``nv`` (float32 [1,1,R,1] tile, per-row valid entry counts: masked on the global ids) or ``vl`` (uint32 [1,1,1,1] row-major, valid slots of this column, >= topk).
    -> (ids uint32 [R,1,1,k] row-major GLOBAL entry ids, descending score; vals bf16 [1,1,R,k] or None). Everything after the local scores lives in the [1,1,R,.] layout (R rows share tile rows).
    """
    from models.demos.blackhole.deepseek_v41_flash.tt.indexer import _cs_vl_full

    k, cols = ix.topk, ix.cols
    gid, col_id = consts["gid"], consts["col_id"]
    sr = ttnn.reshape(scores, [1, 1, R, n])  # row-major: a view
    st_t = None
    if nv is not None:
        g = gid if n == int(gid.shape[3]) else ttnn.slice(gid, [0, 0, 0, 0], [1, 1, 1, n])
        keep = ttnn.lt(g, nv)  # [1,1,R,n]
        st_t = ttnn.typecast(
            ttnn.where(keep, ttnn.typecast(ttnn.to_layout(sr, ttnn.TILE_LAYOUT), ttnn.float32), -1e30), ttnn.bfloat16
        )
        sr = ttnn.to_layout(st_t, ttnn.ROW_MAJOR_LAYOUT)
        vl = _cs_vl_full(ix.md, ix.n_alloc, n)
    if _cs_fast() and not want_vals and "col_u32" in consts and n % 32 == 0:
        return cs_merge_fast(ix, consts, ttnn.reshape(sr, [R, 1, 1, n]), R, n, vl), None
    if st_t is None:
        st_t = ttnn.to_layout(sr, ttnn.TILE_LAYOUT)
    lids = ttnn.experimental.topk_large_indices(
        sr, k=k, valid_length_tensor=vl
    )  # [1,1,R,k] local slots, descending score
    lids_t = ttnn.to_layout(ttnn.typecast(lids, ttnn.uint32), ttnn.TILE_LAYOUT)
    vals = ttnn.gather(st_t, 3, index=lids_t)  # local top-k scores [1,1,R,k] bf16
    gids = ttnn.add(
        ttnn.multiply(ttnn.typecast(lids_t, ttnn.float32), float(cols)), col_id
    )  # cols * slot + c (float32, exact)
    # ``ttnn.gather`` of 32-bit data costs ~2.7 ms (bf16: ~0.1 ms): the global id travels as three bf16-exact 7-bit digits (ids < 2**21), stacked with the scores on the row axis so ONE all_gather carries everything
    hi = ttnn.floor(ttnn.multiply(gids, 1.0 / 16384))
    rem = ttnn.subtract(gids, ttnn.multiply(hi, 16384.0))
    mid = ttnn.floor(ttnn.multiply(rem, 1.0 / 128))
    lo = ttnn.subtract(rem, ttnn.multiply(mid, 128.0))
    planes = ttnn.concat([vals] + [ttnn.typecast(t, ttnn.bfloat16) for t in (hi, mid, lo)], dim=2)  # [1,1,4R,k]
    cand = ix.mc.allgather(ttnn.to_layout(planes, ttnn.ROW_MAJOR_LAYOUT), ix.ccl, axis=1, dim=3)  # [1,1,4R,cols*k]
    W = cand.shape[3]
    cv = ttnn.slice(cand, [0, 0, 0, 0], [1, 1, R, W])
    pos = ttnn.experimental.topk_large_indices(cv, k=k)  # [1,1,R,k]
    pos_t = ttnn.to_layout(ttnn.typecast(pos, ttnn.uint32), ttnn.TILE_LAYOUT)
    parts = ttnn.to_layout(ttnn.slice(cand, [0, 0, R, 0], [1, 1, 4 * R, W]), ttnn.TILE_LAYOUT)
    g3 = ttnn.gather(parts, 3, index=ttnn.concat([pos_t, pos_t, pos_t], dim=2))  # [1,1,3R,k]
    f = lambda i: ttnn.typecast(ttnn.slice(g3, [0, 0, i * R, 0], [1, 1, (i + 1) * R, k]), ttnn.float32)
    ids = ttnn.add(ttnn.add(ttnn.multiply(f(0), 16384.0), ttnn.multiply(f(1), 128.0)), f(2))
    ids = ttnn.reshape(ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [R, 1, 1, k])
    if not want_vals:
        return ids, None
    fv = ttnn.gather(ttnn.to_layout(cv, ttnn.TILE_LAYOUT), 3, index=pos_t)
    return ids, ttnn.to_layout(fv, ttnn.ROW_MAJOR_LAYOUT)


class ColShardSelect:
    def __init__(self, loc, rows_users, ccl, mesh_config, n_global_alloc, topk=512):
        """``loc``: DSV41DecodeIndexer / SpecIndexer built with n_alloc = n_global_alloc / cols (its project / score are reused as they are); ``rows_users``: R = number of query rows per mesh row."""
        md = loc.md
        self.loc, self.md, self.R, self.topk = loc, md, rows_users, topk
        self.rows, self.cols = tuple(md.shape)
        self.n_local = loc.n_alloc
        assert self.n_local * self.cols == n_global_alloc and self.n_local % 32 == 0
        self.ccl, self.mc = ccl, mesh_config
        c = torch.arange(self.cols, dtype=torch.float32)
        j = torch.arange(self.n_local, dtype=torch.float32)
        gid = (j[None, :] * self.cols + c[:, None]).reshape(self.cols, 1, 1, self.n_local)
        self.csh = ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=(self.rows, self.cols))
        up = lambda t, dt, lay: ttnn.from_torch(
            t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self.csh
        )
        self.gid = up(gid, ttnn.float32, ttnn.TILE_LAYOUT)  # [1,1,1,n_local] per device: global id of every local slot
        self.col_id = up(c.reshape(self.cols, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT)
        self.vl_full = up(
            torch.full((self.cols, 1, 1, 1), self.n_local, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self._up = up

    # ---- state -------------------------------------------------------------------------------------------------------------
    def load_keys(self, index_k):
        """index_k [B, N, 128] -> local slabs [U,1,n_local,128] (bfp8_b) per device: user rows over the mesh rows, entry e -> column e % cols, slot e // cols."""
        B, N, _ = index_k.shape
        cols, nl = self.cols, self.n_local
        full = torch.zeros(B, nl * cols, 128, dtype=torch.bfloat16)
        full[:, :N] = index_k.to(torch.bfloat16)
        host = full.reshape(B, nl, cols, 128).permute(0, 2, 1, 3).contiguous()  # [B, cols, nl, 128]
        self.loc.k_cache = ttnn.from_torch(
            host,
            device=self.md,
            dtype=self.loc.key_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, 1), mesh_shape=(self.rows, cols)),
        )

    def valid_per_column(self, n_valid):
        """host uint32 tensor [cols,1,1,1] sharded over the columns: number of valid slots of every column for n_valid valid entries (>= topk, as the replicated ``valid``)."""
        v = torch.tensor([max(-(-(n_valid - c) // self.cols), self.topk) for c in range(self.cols)], dtype=torch.int32)
        h = ttnn.from_torch(
            v.reshape(self.cols, 1, 1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.csh
        )
        return ttnn.to_device(h, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def write_index(self, ent):
        """ent: int32 [rows, cols, U] host (-1 = skip, see ``ent_to_local``) -> device int32 [U] per device (``update_idxs_tensor`` of ``paged_update_cache``)."""
        h = ttnn.from_torch(
            ent.reshape(-1).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.md, dim=0),
        )
        return ttnn.to_device(h, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def ent_to_local(ent, rows, cols):
        """ent int [rows, U] (entry per row / user, <0 = skip) -> int32 [rows, cols, U] per column local slots (owner column: e // cols, others -1)."""
        out = torch.full((rows, cols, ent.shape[1]), -1, dtype=torch.int32)
        for c in range(cols):
            own = (ent >= 0) & (ent % cols == c)
            out[:, c] = torch.where(own, ent // cols, torch.full_like(ent, -1)).to(torch.int32)
        return out

    # ---- selection ---------------------------------------------------------------------------------------------------------
    def select(self, scores, st, vl_col=None, want_vals=False):
        """scores [R,1,1,n_local] row-major bf16 (``loc.score``) -> (ids uint32 [R,1,1,k] row-major (GLOBAL entry ids, descending score), vals bf16 [1,1,R,k] or None unless ``want_vals``).
        ``st["nvalid"]`` (optional float32 [R,1,1,1] tile, per-row valid entry counts): masked on the global ids; otherwise ``vl_col`` (``valid_per_column``) bounds the local top-k. See ``cs_select``.
        """
        R, n = self.R, self.n_local
        if "nvalid" in st:
            nv = st["nvalid_flat"] if "nvalid_flat" in st else ttnn.reshape(st["nvalid"], [1, 1, R, 1])
            vl = None
        else:
            nv, vl = None, vl_col
        self.n_alloc = self.n_local
        from models.demos.blackhole.deepseek_v41_flash.tt.indexer import _cs_consts

        return cs_select(self, _cs_consts(self.md, self.n_local), scores, R, n, nv, vl, want_vals)
