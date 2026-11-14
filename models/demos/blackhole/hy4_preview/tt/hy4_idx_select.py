# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode DSA indexer select pipeline of the integrated Hy4 path (task integ): local scores [1,1,S,n] of THIS column -> blockwise top-k -> multi-core candidate / gather kernels -> all-gather over the
columns -> global top-2048 ids (natural token positions, uint32 [S,1,1,2048], identical on all columns). Copied from probes/idx_1m/idx_pipe.py (idx-1m, exact up to bf16 ties); the all-gather is injected
(``ag(t)``: the model's CCL manager) and the kernels live in ``hy4_idx_cs_kernels`` (LAYOUT=1: 512-token pages striped over 8 columns in 64-token slices).
"""
import os

import ttnn

COLS, TOPK = 8, 2048
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_idx_cs_kernels")

def _cb(core_set, idx, pages, page, dt):
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
    )


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _hash(tag, *parts):
    return (tag << 40) | (hash(parts) & ((1 << 40) - 1))


def _prog(name, mesh, rows, cbs_spec, ct, accs, common, tensors, tag, ncores=None):
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][: min(rows, grid.x * grid.y, ncores or 10**9)]
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


def topk_ids(x, k, valid=None):
    ids = ttnn.experimental.topk_large_indices(x, k=k, **({"valid_length": valid} if valid else {}))
    return ids if ids.dtype == ttnn.uint32 else ttnn.typecast(ids, ttnn.uint32)


# ------------------------------------------------------------------ v2: blockwise local stage, multi-core candidate / gather kernels
def cs_cand2(sc, ids, colid, layout=1, blks=0, nblk=1, split=1, ncores=None):
    """sc bf16 RM [R,1,1,N], ids u32 RM [R,1,1,K] -> (vals bf16 [R,1,1,K], gids u32 [R,1,1,K]); R*split work items (cores)."""
    R, N, K = int(sc.shape[0]), int(sc.shape[-1]), int(ids.shape[-1])
    mesh = sc.device()
    KP = K // split
    vals = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    gids = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    u32, bf = ttnn.uint32, ttnn.bfloat16
    _prog(
        "cs_cand2.cpp",
        mesh,
        R * split,
        [(1, KP * 4, u32), (1, KP * 64, u32), (1, KP * 2, bf), (1, KP * 4, u32), (1, 64, u32)],
        [K, COLS, N, layout, blks, nblk, split],
        [sc, ids, colid, vals, gids],
        [
            sc.buffer_address(),
            ids.buffer_address(),
            colid.buffer_address(),
            vals.buffer_address(),
            gids.buffer_address(),
        ],
        [sc, ids, colid, vals, gids],
        0xC61,
        ncores,
    )
    return vals, gids


def cs_gather(pos, cid, cval=None, split=1, ncores=None):
    """pos u32 [R,1,1,K], cid u32 [R,1,1,W] (, cval bf16 [R,1,1,W]) -> (ids u32 [R,1,1,K], vals bf16 or None), top-k order, no sort."""
    R, K, W = int(pos.shape[0]), int(pos.shape[-1]), int(cid.shape[-1])
    mesh = pos.device()
    KP = K // split
    want = cval is not None
    oid = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    oval = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([R, 1, 1, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        if want
        else oid
    )
    cv = cval if want else cid
    u32, bf = ttnn.uint32, ttnn.bfloat16
    _prog(
        "cs_gather.cpp",
        mesh,
        R * split,
        [
            (1, KP * 4, u32),
            (1, W * 4, u32),
            (1, W * 2 if want else 64, bf),
            (1, KP * 4, u32),
            (1, KP * 2 if want else 64, bf),
        ],
        [K, W, split, 1 if want else 0],
        [pos, cid, cv, oid, oval],
        [pos.buffer_address(), cid.buffer_address(), cv.buffer_address(), oid.buffer_address(), oval.buffer_address()],
        [pos, cid, cv, oid, oval],
        0xC62 + (1 if want else 0),
        ncores,
    )
    return oid, (oval if want else None)


def select_v2(ag, colid, rows, n, B=16, kb=512, kloc=1024, K=TOPK, split=4, sort_final=False, max_id=None):
    """rows bf16 RM [1,1,S,n] local score rows -> ids u32 [S,1,1,K] global. Local stage = blockwise top-k (B blocks, kb each) + local merge to kloc; then AG + merge to K."""
    S = int(rows.shape[2])
    m = n // B
    blk = ttnn.reshape(rows, [1, 1, S * B, m])
    bids = topk_ids(blk, kb, m)
    bvals, bgids = cs_cand2(
        ttnn.reshape(rows, [S * B, 1, 1, m]), ttnn.reshape(bids, [S * B, 1, 1, kb]), colid, 1, m, B, split
    )
    cvals = ttnn.reshape(bvals, [1, 1, S, B * kb])
    pos = topk_ids(cvals, kloc)
    lids, lvals = cs_gather(
        ttnn.reshape(pos, [S, 1, 1, kloc]),
        ttnn.reshape(bgids, [S, 1, 1, B * kb]),
        ttnn.reshape(bvals, [S, 1, 1, B * kb]),
        split,
    )
    av = ag(lvals)
    ai = ag(lids)
    W = int(av.shape[-1])
    pos2 = ttnn.reshape(topk_ids(ttnn.reshape(av, [1, 1, S, W]), K), [S, 1, 1, K])
    if sort_final:
        return cs_final(pos2, ai, max_id or n * COLS)
    return cs_gather(pos2, ai, None, split)[0]
