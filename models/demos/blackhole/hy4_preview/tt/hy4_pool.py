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


def slice_copy(src, dst, colid, units, page, npages, dst_page0=0, ncores=32):
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
        kernel_source=f"{KDIR}/slice_copy.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[src.buffer_address(), dst.buffer_address(), colid.buffer_address(), npages, dst_page0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, units * page + 64)])
    prog.custom_program_hash = _hash(0x9B2, units, page, nc, tuple(ct))
    return ttnn.generic_op([src, dst, colid], prog)


def id_filter(gids, colinfo, S, owners=COLS, out=None):
    """gids uint32 RM [S,1,1,K] global ids (colinfo: per-device uint32 page, element 0 = column, element 1 = valid token count n: ids >= n are dropped; sentinel 0xFFFFFFFF is >= n) -> idx8 uint32 RM [1,1,owners*S,K] local rows of THIS column (sentinel tail), rows (j, s) = j*S + s."""
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
    ct = [K, S, owners, 0] + _acc(gids) + _acc(colinfo) + _acc(out)
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


# ------------------------------------------------------------------ host geometry
def locate(t):
    """token position -> (page, column, local row)."""
    t = np.asarray(t, dtype=np.int64)
    return t // CHUNK, (t % CHUNK) // SLICE, (t // CHUNK) * SLICE + t % SLICE


def write_rows(pos, rows=4):
    """pos [rows] -> int32 [rows, COLS, 1, 16]: element 0 = local row of the appended token on its owner column, -1 (0xFFFFFFFF) elsewhere."""
    out = torch.full((rows, COLS, 1, 16), -1, dtype=torch.int32)
    for r, p in enumerate(pos):
        _, c, loc = locate(int(p))
        out[r, int(c), 0, 0] = int(loc)
    return out


def key_counts(pos, rows=4):
    """valid local keys per column AFTER appending the token at ``pos``: int32 [rows, COLS, 1, 16] (element 0)."""
    out = torch.zeros((rows, COLS, 1, 16), dtype=torch.int32)
    for r, p in enumerate(pos):
        n = int(p) + 1
        full, rem = divmod(n, CHUNK)
        for c in range(COLS):
            out[r, c, 0, 0] = full * SLICE + min(max(rem - SLICE * c, 0), SLICE)
    return out


def colid_host(rows=4, nvalid=None):
    """[rows, COLS, 1, 16] int32: element 0 = column id, element 1 = valid token count of the row's user (``nvalid`` [rows], default 2**30)."""
    c = torch.zeros(rows, COLS, 1, 16, dtype=torch.int32)
    c[:, :, 0, 0] = torch.arange(COLS)[None, :]
    c[:, :, 0, 1] = 1 << 30 if nvalid is None else torch.as_tensor(nvalid, dtype=torch.int32)[:, None]
    return c
