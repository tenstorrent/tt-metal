# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-sharded SCALED_FP8 KV pool of the Hy4 long-context decode path (task integ; kv-design/DESIGN.md).

Geometry (per mesh row = one user at B=4): page = 512 tokens striped over the 8 mesh columns in 64-token slices: column c holds tokens [p*512 + 64c, p*512 + 64c + 64) of page p at local
rows [64p, 64p + 64) (identity page table: pool page p == page p of the user's sequence; a page table is only needed for prefix-cache sharing, a later step). Per layer and per device:
  kv pool    fp8_e4m3 ROW_MAJOR [1,1,P*64,656]   (SCALED_FP8 packed rows: 512 e4m3 + 4 fp32 scales + 64 bf16 rope, stored at the 64 B aligned stride 704)
  key pool   bfp8_b  TILE      [P,1,64,128]      (indexer keys of the 21 full-indexer layers + MTP; viewed as [1,1,P*64,128] by the score kernel)
JIT kernels in ``hy4_pool_kernels`` (no C++ / image change): ``row_write`` (decode append: one row to a data-dependent local row, skipped on non-owner columns), ``slice_copy`` (natural-order cache ->
column slices of the pool: prefill hand-off and synthetic fill), ``id_filter`` (global top-k ids -> this column's local rows, sentinel tail, replicated to the 8 owner blocks).
"""
import os

import numpy as np
import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_pool_kernels")
CHUNK, COLS, SLICE = 512, 8, 64
SENT = 0xFFFFFFFF
ROWB = 704  # aligned stride of a 656-byte SCALED_FP8 row
KEYTILE = 1088  # bfp8_b tile page


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _hash(tag, *parts):
    return (tag << 40) | (hash(parts) & ((1 << 40) - 1))


def _cores(n, gx):
    return [(i % gx, i // gx) for i in range(n)]


def _core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _cb(core_set, total):
    return ttnn.CBDescriptor(
        total_size=total,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )


def row_write(src, ids, dst, n=1, page=ROWB):
    """In place: dst[ids[i]] = src[i], i < n (ids int32/uint32 [.,16] per device; 0xFFFFFFFF = skip). src/dst row-major with the same row stride ``page`` bytes (704 for SCALED_FP8)."""
    cs = _core_set([(0, 0)])
    idp = 64
    ct = [n, 0, page, idp] + _acc(src) + _acc(ids) + _acc(dst)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/row_write.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[src.buffer_address(), ids.buffer_address(), dst.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, idp + page)])
    prog.custom_program_hash = _hash(0x9B1, n, page, tuple(ct))
    return ttnn.generic_op([src, ids, dst], prog)


def slice_copy(src, dst, colid, units, page, npages, dst_page0=0, ncores=32, kernel="slice_copy.cpp"):
    """natural-order ``src`` pages -> this column's slices in ``dst`` (see slice_copy.cpp). units = 64 (rows) or 8 (key tiles); page = bytes per page (704 rows, 1088 bfp8 tiles, 2048 bf16 tiles).
    npages = number of 512-token pool pages to copy, written at pool page ``dst_page0``."""
    md = dst.device()
    grid = md.compute_with_storage_grid_size()
    nc = max(1, min(ncores, npages))
    cores = _cores(nc, grid.x)
    cs = _core_set(cores)
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i]
    ct = [units, page, nc, 0] + _acc(src) + _acc(dst) + _acc(colid)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{kernel}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[src.buffer_address(), dst.buffer_address(), colid.buffer_address(), npages, dst_page0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, units * page + 64)])
    prog.custom_program_hash = _hash(0x9B2, units, page, nc, tuple(ct), kernel)
    return ttnn.generic_op([src, dst, colid], prog)


def id_filter(gids, colinfo, S, owners=COLS, out=None, perrow=False):
    """gids uint32 RM [S,1,1,K] global ids (colinfo: per-device uint32 page, element 0 = column, element 1 = valid token count n: ids >= n are dropped; sentinel 0xFFFFFFFF is >= n; perrow: element 2+s = n of row s, verify rows) -> idx8 uint32 RM [1,1,owners*S,K] local rows of THIS column (sentinel tail), rows (j, s) = j*S + s."""
    md = gids.device()
    K = int(gids.shape[-1])
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, owners * S, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
    grid = md.compute_with_storage_grid_size()
    cores = _cores(S, grid.x)
    cs = _core_set(cores)
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i]
    ct = [K, S, owners, 0, int(perrow)] + _acc(gids) + _acc(colinfo) + _acc(out)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/id_filter.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[gids.buffer_address(), colinfo.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, 2 * K * 4 + 64)])
    prog.custom_program_hash = _hash(0x9B3, K, S, owners, tuple(ct))
    ttnn.generic_op([gids, colinfo, out], prog)
    return out


def push_wait(push, S):
    """B35: consumer half of a hy4_qpush.Push of bf16 RM rows [S,1,1,PB/2]: waits for the per-source flags and assembles [S,1,1,8*PB/2] exactly like all_gather_async(dim=3)."""
    md = push.md
    PB, RP, NP = push.pb, push.rp, push.np
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, 1, 1, 8 * PB // 2]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = md.compute_with_storage_grid_size()
    cores = _cores(S, grid.x)
    cs = _core_set(cores)
    fx, fy = push._phys(*push.fcore)
    extra = [fx, fy] + list(push.flag_addrs) + [push.cnt_addr]
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i] + extra
    ct = [S, 0, PB, RP, NP] + _acc(push.land) + _acc(out)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/pw_wait.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[push.land.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, 8 * PB + 128)])
    prog.custom_program_hash = _hash(0x9B7, S, PB, RP, NP, tuple(ct))
    ttnn.generic_op([push.land, out], prog)
    return out


def gather_filter_own(pos, lids, colinfo, S, out, perrow=False, nq=None):
    """B35: ``cs_gather`` + ``id_filter`` + the id all-gather as one multi-core program (nq cores per row), bit-identical to the stock chain given the layout invariant (candidate block c of the gathered row holds tokens of column c): pos u32 RM [S,1,1,K], lids u32 RM [S,1,1,KL]
    (this chip's local candidate ids), colinfo / out as ``id_filter``."""
    md = pos.device()
    K, KL = int(pos.shape[-1]), int(lids.shape[-1])
    nq = nq or int(os.environ.get("HY4_B35_NQ", "8"))
    assert K % nq == 0
    grid = md.compute_with_storage_grid_size()
    assert nq <= grid.x and S <= grid.y
    cores = [(q, s) for s in range(S) for q in range(nq)]
    cs = _core_set(cores)
    rt = ttnn.RuntimeArgs()
    for s in range(S):
        lc = md.worker_core_from_logical_core(ttnn.CoreCoord(0, s))
        for q in range(nq):
            rt[q][s] = [s, q, lc.x, lc.y]
    total = (K // nq) * 4 * (1 + nq) + K * 4 + KL * 4 + 64 + nq * 16
    ct = [K, S, COLS, 0, int(perrow), KL, nq] + _acc(pos) + _acc(lids) + _acc(colinfo) + _acc(out)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/gf_own.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[
            pos.buffer_address(),
            lids.buffer_address(),
            colinfo.buffer_address(),
            out.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[k], semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)], cbs=[_cb(cs, total)]
    )
    prog.custom_program_hash = _hash(0x9B6, K, S, nq, tuple(ct))
    ttnn.generic_op([pos, lids, colinfo, out], prog)
    return out


_DBG = {}


def dbg_dump(md):
    """sel3x debug counters (HY4_B36_DBG=1): per device aggregate over its (row, core) pages: [calls, N>K calls, sum g, max g, sum long runs, max run, sum runs, sum jmax, max jmax]"""

    d = _DBG.get(id(md))
    if d is None:
        return None
    out = []
    for i, t in enumerate(ttnn.get_device_tensors(d)):
        x = ttnn.to_torch(t).reshape(-1, 16).long()
        out.append(
            [
                int(x[:, 0].max()),
                int(x[:, 1].max()),
                int(x[:, 2].sum()),
                int(x[:, 3].max()),
                int(x[:, 4].sum()),
                int(x[:, 5].max()),
                int(x[:, 6].sum()),
                int(x[:, 7].sum()),
                int(x[:, 8].max()),
            ]
        )
    return out


def sel3_own(push, lids, colinfo, S, out, perrow=False, K=2048, nq=None):
    """B36 (HY4_B36_SEL3=1): stage-3 select of the indexer as one flag-consuming program. ``push`` = hy4_qpush.Push of the bf16 candidate score rows [S,1,1,KL] (already launched; every row sorted descending, as the stage-2 top-k
    output), ``lids`` u32 [S,1,1,KL] this chip's local candidate ids (same order), colinfo / out as ``gather_filter_own``. Replaces push_wait + the stock top-K over the 8*KL gathered scores + gather_filter_own.
    """
    md = push.md
    KL, PB, RP, NP = int(lids.shape[-1]), push.pb, push.rp, push.np
    assert PB == KL * 2
    nq = nq or int(os.environ.get("HY4_B36_NQ", "8"))
    grid = md.compute_with_storage_grid_size()
    assert nq <= grid.x and S <= grid.y and KL % nq == 0
    cores = [(q, s) for s in range(S) for q in range(nq)]
    cs = _core_set(cores)
    fx, fy = push._phys(*push.fcore)
    extra = [fx, fy] + list(push.flag_addrs) + [push.cnt_addr]
    rt = ttnn.RuntimeArgs()
    for s in range(S):
        lc = md.worker_core_from_logical_core(ttnn.CoreCoord(0, s))
        for q in range(nq):
            rt[q][s] = [s, q, lc.x, lc.y] + extra
    kq = KL // nq
    exact = os.environ.get("HY4_B36_EXACT", "0") == "1"  # stock tie order emulated (sel3x.cpp)
    total = 8 * PB + 128 + kq * 4 + 64 + nq * kq * 4 + nq * 16 + K * 4
    if exact:
        total = 8 * PB + 128 + KL * 4 + 64 + nq * KL * 4 + nq * 16 + K * 4 + KL * 4 + 8 * KL * 2
    dbg = None
    if (
        exact and os.environ.get("HY4_B36_DBG", "0") == "1"
    ):  # debug counters of sel3x (tie statistics of the real data), read back by dbg_dump
        dbg = _DBG.get(id(md))
        if dbg is None:
            import torch

            dbg = _DBG[id(md)] = ttnn.from_torch(
                torch.zeros(S * nq, 1, 1, 16, dtype=torch.int32),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
    ct = (
        [K, S, COLS, 0, int(perrow), KL, nq, PB, RP, NP]
        + _acc(push.land)
        + _acc(lids)
        + _acc(colinfo)
        + _acc(out)
        + _acc(dbg if dbg is not None else out)
    )
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/sel3x.cpp" if exact else f"{KDIR}/sel3.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[
            push.land.buffer_address(),
            lids.buffer_address(),
            colinfo.buffer_address(),
            out.buffer_address(),
            dbg.buffer_address() if dbg is not None else 0,
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[k], semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)], cbs=[_cb(cs, total)]
    )
    prog.custom_program_hash = _hash(0x9B8, K, S, nq, int(exact), tuple(ct))
    ttnn.generic_op([push.land, lids, colinfo, out], prog)
    return out


def sel2(bvals, bgids, S, nb, kb, kl, blocked, nq=None):
    """B36 (HY4_B36_SEL2=1): stage-2 select of the indexer, one program: the kl best of the nb sorted candidate lists (cs_cand2 outputs ``bvals`` bf16 / ``bgids`` u32: blocked layout [S,1,1,nb*kb], else [S*nb,1,1,kb]) in total
    order (value desc, list asc, position asc); returns (lids u32 [S,1,1,kl], lvals bf16 [S,1,1,kl]), sorted descending, exhausted entries (-inf, 0xFFFFFFFF). Replaces the stock top-kl + cs_gather.
    """
    md = bvals.device()
    nq = nq or int(os.environ.get("HY4_B36_NQ2", "8"))
    grid = md.compute_with_storage_grid_size()
    assert nq <= grid.x and S <= grid.y and kl % nq == 0 and nb <= 16
    lv = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, 1, 1, kl]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    li = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, 1, 1, kl]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = [(q, s) for s in range(S) for q in range(nq)]
    cs = _core_set(cores)
    rt = ttnn.RuntimeArgs()
    for s in range(S):
        for q in range(nq):
            rt[q][s] = [s, q]
    oq = kl // nq
    total = nb * kb * 2 + oq * 64 + ((oq * 2 + 63) // 64) * 64 + oq * 4 + 64
    ct = [S, nq, kl, nb, kb, int(blocked), 0] + _acc(bvals) + _acc(bgids) + _acc(lv) + _acc(li)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/sel2.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[
            bvals.buffer_address(),
            bgids.buffer_address(),
            lv.buffer_address(),
            li.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, total)])
    prog.custom_program_hash = _hash(0x9B9, S, nq, nb, kb, kl, int(blocked), tuple(ct))
    ttnn.generic_op([bvals, bgids, lv, li], prog)
    return li, lv


def gather_filter(pos, cand, colinfo, S, out, perrow=False, push=None, own=False):
    """B35: ``cs_gather`` + ``id_filter`` as one program (bit-identical). pos u32 RM [S,1,1,K] (top-k positions into the W candidates), cand u32 RM [S,1,1,W] (all-gathered ids) or, with ``push`` (a hy4_qpush.Push of the local id
    rows, already launched), its landing tensor (the program waits for the per-source flags). colinfo / out as ``id_filter``.
    """
    md = pos.device()
    K = int(pos.shape[-1])
    owners = COLS
    grid = md.compute_with_storage_grid_size()
    cores = _cores(S, grid.x)
    cs = _core_set(cores)
    rt = ttnn.RuntimeArgs()
    if push is None:
        W = int(cand.shape[-1])  # own=True: cand = this chip's local ids [S,1,1,KL] (W = KL)
        land, PB, RP, NP = (2 if own else 0), 0, 1, 1
        extra = []
    else:
        PB, RP, NP = push.pb, push.rp, push.np
        W = COLS * PB // 4
        land = 1
        fx, fy = push._phys(*push.fcore)
        extra = [fx, fy] + list(push.flag_addrs) + [push.cnt_addr]
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i] + extra
    total = 2 * K * 4 + W * 4 + 64 + 128
    ct = [K, S, owners, 0, int(perrow), W, land, PB, RP, NP] + _acc(pos) + _acc(cand) + _acc(colinfo) + _acc(out)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/gf_filter.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[
            pos.buffer_address(),
            cand.buffer_address(),
            colinfo.buffer_address(),
            out.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, total)])
    prog.custom_program_hash = _hash(0x9B5, K, S, owners, tuple(ct))
    ttnn.generic_op([pos, cand, colinfo, out], prog)
    return out


# ------------------------------------------------------------------ host geometry
def locate(t):
    """token position -> (page, column, local row)."""
    t = np.asarray(t, dtype=np.int64)
    return t // CHUNK, (t % CHUNK) // SLICE, (t // CHUNK) * SLICE + t % SLICE


def write_rows(pos, rows=4, S=1):
    """pos [rows] -> int32 [rows, COLS, 1, 16]: element i (i < S verify rows, token pos+i) = local row of the appended token on its owner column, -1 (0xFFFFFFFF) elsewhere."""
    out = torch.full((rows, COLS, 1, 16), -1, dtype=torch.int32)
    for r, p in enumerate(pos):
        for i in range(S):
            _, c, loc = locate(int(p) + i)
            out[r, int(c), 0, i] = int(loc)
    return out


def key_counts(pos, rows=4, S=1):
    """valid local keys per column AFTER appending the token at ``pos``: int32 [rows, COLS, 1, 16] (element 0). S verify rows: element j = the count for the query row at pos+j (it sees the tokens up to pos+j)."""
    out = torch.zeros((rows, COLS, 1, 16), dtype=torch.int32)
    for r, p in enumerate(pos):
        for j in range(S):
            n = int(p) + j + 1
            full, rem = divmod(n, CHUNK)
            for c in range(COLS):
                out[r, c, 0, j] = full * SLICE + min(max(rem - SLICE * c, 0), SLICE)
    return out


def colid_host(rows=4, nvalid=None, S=1):
    """[rows, COLS, 1, 16] int32: element 0 = column id, element 1 = valid token count of the row's user (``nvalid`` [rows], default 2**30); S > 1 verify rows: element 2+j = nvalid + j (row j sees the tokens up to its own position)."""
    c = torch.zeros(rows, COLS, 1, 16, dtype=torch.int32)
    c[:, :, 0, 0] = torch.arange(COLS)[None, :]
    nv = (
        torch.full((rows,), 1 << 30, dtype=torch.int32)
        if nvalid is None
        else torch.as_tensor(nvalid, dtype=torch.int32)
    )
    c[:, :, 0, 1] = nv[:, None]
    for j in range(S if S > 1 else 0):
        c[:, :, 0, 2 + j] = (nv + j)[:, None] if nvalid is not None else (1 << 30)
    return c


def colid_masked_host(skip_rows, rows=4):
    """colid page with element 2 = 1 on the mesh rows in ``skip_rows`` (slice_copy_masked.cpp copies nothing there)."""
    c = colid_host(rows)
    for r in skip_rows:
        c[int(r), :, 0, 2] = 1
    return c
