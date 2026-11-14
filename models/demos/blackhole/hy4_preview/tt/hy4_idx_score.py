# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-specialised DSA indexer score (task fuse-sparse-merge, B1; flag HY4_FUSE=idx), ONE generic_op program, JIT kernels (no image rebuild).

score[r, t] = sum_h w[r,h] * relu(q[r,h] . k[t])  for S <= 16 query rows (decode users / verify rows), keys t < kv_len, -inf at t >= kv_len.
The stock ``indexer_score_dsa`` pads the query rows to a 32-row tile and runs 32 heads x 4 matmul tile ops per key tile (25x too many for S=1); here the HEADS are the M rows:
  phase A: relu(Q_r [32 heads x 128] @ K^T) = 4 matmul tile ops per (key tile, r), packed with relu;  phase B: w_r [row r] @ qk = one matmul tile op, accumulated over r.
Inputs:  q [S,1,32,128] bf16 TILE (head h = tile row h of query row s: ``q_headrows``), keys [1,1,n,128] bf8 TILE (n % 32 == 0), w = the stock weights [1,1,32,32] bf16 TILE (row r = query row r, col = head).
Output:  [1,1,S,n] bf16 ROW_MAJOR (what the stock op returns for S rows), consumed by topk_large_indices.
"""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_idx_kernels")


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def q_headrows(q, S):
    """stock-layout q [1,32,S,128] (head h, query row s) -> [S,1,32,128] TILE (tile row = head): 2 stock ops, only for tests/bring-up; the model should emit this layout from the q projection directly."""
    x = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.permute(x, (2, 0, 1, 3))  # [S,1,32,128]
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def idx_score(q, keys, w, kv_len, S, ncores=None, kc=None, out=None, fid=None, kvt=None, n_rows=None, kvt_rows=False):
    """kvt (optional): uint32/int32 ROW_MAJOR [.,16] per-device tensor whose element 0 is the number of valid keys of THIS device (trace-safe; kv_len is then ignored).
    kvt_rows=True (verify rows, S <= 16): element r is the number of valid keys of query row r (row r sees the keys up to its own position: causal among the verify rows).
    """
    md = keys.device()
    n = int(
        n_rows or keys.shape[2]
    )  # n_rows: score only the first n_rows key rows of a larger pool tensor (decode bucket)
    assert n % 32 == 0 and S <= 16
    R = n // 32
    grid = md.compute_with_storage_grid_size()
    ncores = ncores or int(os.environ.get("HY4_IDX_NCORES", str(grid.x * grid.y)))
    ncores = min(ncores, R)
    KC = kc or int(os.environ.get("HY4_IDX_KC", str(max(1, 8 // S))))
    assert KC * S <= 8
    base, extra = divmod(R, ncores)
    cores, rows = [], []
    r0 = 0
    for i in range(ncores):
        nr = base + (1 if i < extra else 0)
        cores.append((i % grid.x, i // grid.x))
        rows.append((r0, nr))
        r0 += nr
    maxr = base + (1 if extra else 0)
    rngs, nleft, r = [], ncores, 0
    while nleft > 0:
        wd = min(nleft, grid.x)
        rngs.append(ttnn.CoreRange(ttnn.CoreCoord(0, r), ttnn.CoreCoord(wd - 1, r)))
        nleft -= wd
        r += 1
    core_set = ttnn.CoreRangeSet(rngs)
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, S, n]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
    Q, W, K, TMP, QK, OUT, STRIP, KVCB = range(8)
    bf = ttnn.bfloat16
    KT = 1088

    def cb(i, n_, page, df):
        return ttnn.CBDescriptor(
            total_size=page * n_,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=df, page_size=page)],
        )

    cbs = [
        cb(Q, S * 4, 2048, bf),
        cb(W, S, 2048, bf),
        cb(K, 2 * KC * 4, KT, ttnn.bfloat8_b),
        cb(TMP, 1, 2048, bf),
        cb(QK, S * KC, 2048, bf),
        cb(OUT, 2 * KC, 2048, bf),
        ttnn.CBDescriptor(
            total_size=S * maxr * 64,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=STRIP, data_format=bf, page_size=S * maxr * 64)],
        ),
        cb(KVCB, 1, 64, ttnn.int32),
    ]
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), (a, nr) in zip(cores, rows):
        rt_r[cx][cy] = [a, nr]
        rt_w[cx][cy] = [a, nr]
        rt_c[cx][cy] = [nr]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/isc_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[S, KC, Q, W, K, TMP, KT] + _acc(keys) + _acc(q) + _acc(w),
        runtime_args=rt_r,
        common_runtime_args=[keys.buffer_address(), q.buffer_address(), w.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/isc_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[S, KC, OUT, STRIP, n * 2, maxr, (2 if kvt_rows else 1) if kvt is not None else 0, KVCB]
        + _acc(out)
        + (_acc(kvt) if kvt is not None else []),
        runtime_args=rt_w,
        common_runtime_args=[out.buffer_address(), int(kv_len) if kvt is None else 0]
        + ([kvt.buffer_address()] if kvt is not None else []),
        config=ttnn.WriterConfigDescriptor(),
    )
    fid = (
        fid
        or {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[
            os.environ.get("HY4_IDX_FID", "hifi2")
        ]
    )
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False)
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/isc_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[S, KC, Q, W, K, QK, OUT],
        runtime_args=rt_c,
        common_runtime_args=[],
        config=cc,
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x1D5 << 40) | (
        hash(
            (
                S,
                n,
                ncores,
                KC,
                str(fid),
                int(kvt is not None),
                int(kvt_rows),
                tuple(_acc(keys)),
                tuple(_acc(q)),
                tuple(_acc(w)),
                tuple(_acc(out)),
            )
        )
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([q, keys, w] + ([kvt] if kvt is not None else []) + [out], prog)
    return out
