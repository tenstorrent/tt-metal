# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""otherlayers: JIT programs (ttnn.generic_op) of the decode indexer / owner-layer glue, see tt/ol_kernels/*.cpp.

idx_qpost(qlin, C, S, R, out_l1)   the RoPE of the indexer queries as ONE program: wq_b output qlin [1,1,R,4096] (heads on the columns) -> [R,1,32,128] bf16 TILE (heads on the rows) rotated like
                                   ttnn.addcmul(ttnn.multiply(q, C), ttnn.matmul(q, P), S) (replaces reshape (20 us) + multiply + matmul + addcmul = 42 us); bit-identical operand construction of the
                                   post-SDPA glue (attn_kernels/post_reader.cpp / post_compute.cpp UNF)."""
import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_pre import _merged_set

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ol_kernels")
ADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_kernels")


def _v2():
    """DSV41_OL_QPOST2=1: the table rows broadcast by the unpacker (qpost2_*.cpp) instead of RISC copy loops"""
    return "2" if os.environ.get("DSV41_OL_QPOST2", "0") == "1" else ""


def _zdefs():
    """DSV41_OL_ZONES=1: compile the profiler zones of the ol_kernels in (ol_zones.h)"""
    return [("OL_ZONES", "1")] if os.environ.get("DSV41_OL_ZONES", "0") == "1" else []


def _kern(path, core_set, ct, rt, cfg, common):
    return ttnn.KernelDescriptor(
        kernel_source=path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common,
        config=cfg,
        defines=_zdefs(),
    )


PFDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pf_kernels")


def keyrows_ok(k, U, n, kcfg):
    return (
        k.dtype == ttnn.bfloat16
        and k.layout == ttnn.TILE_LAYOUT
        and int(k.shape[-1]) == 128
        and int(k.shape[-2]) == U * n
        and U * n <= 16
        and U <= 8
        and kcfg is not None
    )


def key_rows(k, U, n, kcfg):
    """k [1,1,R,128] bf16 TILE (R = U * n key rows after RoPE, row u * n + j = block index j of user u) -> list of n tensors [1,U,1,128] bf16 TILE L1 height-sharded (``kcfg``), the update tiles of
    ``paged_update_cache`` (key in row 0, the rest zero) of the block indices j, the keys fp4-simulated: ONE program replaces pad + fp4_fused + slice + to_layout(RM) + n x (slice, to_layout(TILE),
    to_memory_config). One core per 32-column tile column of the key (the per-tile fp4 blocks are independent), the fp4 math is the pf_kernels/fp4_compute.cpp one.
    """
    assert keyrows_ok(k, U, n, kcfg)
    md = k.device()
    outs = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, U, 1, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, kcfg)
        for _ in range(n)
    ]
    grid = md.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    cores = allc[:4]
    cset = _merged_set(cores)
    bf = ttnn.bfloat16
    CB_X, CB_ONE, CB_ABS, CB_INV, CB_SC, CB_Q, CB_OUT, CB_Z = range(8)
    cbs = [
        _cb(cset, CB_X, 2, 2048, bf),
        _cb(cset, CB_ONE, 1, 2048, bf),
        _cb(cset, CB_ABS, 1, 2048, bf),
        _cb(cset, CB_INV, 1, 2048, bf),
        _cb(cset, CB_SC, 1, 2048, bf),
        _cb(cset, CB_Q, 1, 2048, bf),
        _cb(cset, CB_OUT, 2, 2048, bf),
        _cb(cset, CB_Z, 1, 2048, bf),
    ]
    rt = (
        crt
    ) = (
        wrt
    ) = ttnn.RuntimeArgs()  # no per-core runtime arguments: the tile column is the logical x of the core (ol_core_idx)
    ucores = ttnn.corerange_to_cores(
        ttnn.num_cores_to_corerangeset(U, ttnn.CoreCoord(8, 8), row_wise=True), row_wise=True
    )
    wcommon = [o.buffer_address() for o in outs]
    for cc in ucores[:U]:
        p = md.worker_core_from_logical_core(ttnn.CoreCoord(cc.x, cc.y))
        wcommon += [p.x, p.y]
    gx = int(grid.x)
    wcommon = wcommon + [gx]
    reader = _kern(
        f"{KDIR}/fp4n_reader.cpp",
        cset,
        [CB_X, CB_ONE, 1] + _acc(k),
        rt,
        ttnn.ReaderConfigDescriptor(),
        [k.buffer_address(), gx],
    )
    writer = _kern(
        f"{KDIR}/keyrows_writer.cpp",
        cset,
        [CB_OUT, CB_Z, U, n],
        wrt,
        ttnn.WriterConfigDescriptor(),
        wcommon,
    )
    compute = _kern(
        f"{KDIR}/fp4n_compute.cpp",
        cset,
        [CB_X, CB_ONE, CB_ABS, CB_INV, CB_SC, CB_Q, CB_OUT, 1],
        crt,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x0A2, os.environ.get("DSV41_OL_ZONES", "0"), U, n, tuple(_acc(k)), tuple(wcommon[n:-1])
    )
    ttnn.generic_op([k] + outs, prog)
    return outs


def cs_cand_fast(sc, lids, colid, cols, split=4):
    """tt/indexer_colshard.cs_cand with ol_kernels/cs_cand_fast.cpp: the whole score row in one NOC read, the K ids of a row over ``split`` cores. Same outputs (vals bf16 RM [R,1,1,K], gids uint32 RM [R,1,1,K])."""
    R, n, K = int(sc.shape[0]), int(sc.shape[-1]), int(lids.shape[-1])
    mesh = sc.device()
    assert (
        sc.dtype == ttnn.bfloat16 and sc.layout == ttnn.ROW_MAJOR_LAYOUT and lids.dtype == ttnn.uint32 and n % 32 == 0
    )
    assert K % (split * 32) == 0 and n * 2 <= 98304
    KP = K // split
    vals = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    gids = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 1, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    cores = allc[: R * split]
    cset = _merged_set(cores)
    u32, bf = ttnn.uint32, ttnn.bfloat16
    cbs = [
        _cb(cset, 0, 1, KP * 4, u32),
        _cb(cset, 1, 1, n * 2, bf),
        _cb(cset, 2, 1, KP * 2, bf),
        _cb(cset, 3, 1, KP * 4, u32),
        _cb(cset, 4, 1, 64, u32),
    ]
    rt = ttnn.RuntimeArgs()  # no per-core runtime arguments (ol_core_idx)
    kern = _kern(
        f"{KDIR}/cs_cand_fast.cpp",
        cset,
        [0, 1, 2, 3, 4, K, cols, n, split] + _acc(sc) + _acc(lids) + _acc(colid) + _acc(vals) + _acc(gids),
        rt,
        ttnn.ReaderConfigDescriptor(),
        [
            sc.buffer_address(),
            lids.buffer_address(),
            colid.buffer_address(),
            vals.buffer_address(),
            gids.buffer_address(),
            int(grid.x),
        ],
    )
    prog = ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x0A3,
        os.environ.get("DSV41_OL_ZONES", "0"),
        R,
        K,
        n,
        cols,
        split,
        tuple(_acc(sc)),
        tuple(_acc(lids)),
        tuple(_acc(colid)),
        tuple(_acc(vals)),
        tuple(_acc(gids)),
    )
    ttnn.generic_op([sc, lids, colid, vals, gids], prog)
    return vals, gids


def qpost_ok(qlin, C, S, R):
    return (
        qlin.dtype == ttnn.bfloat16
        and C.dtype == ttnn.bfloat16
        and S.dtype == ttnn.bfloat16
        and qlin.layout == ttnn.TILE_LAYOUT
        and int(qlin.shape[-1]) == 4096
        and int(qlin.shape[-2]) == R
        and R <= 16
    )


def idx_qpost(qlin, C, S, R, out_l1=True):
    """qlin [1,1,R,4096] bf16 TILE (R <= 16 rows in one tile row), C / S [R,1,1,128] bf16 TILE -> [R,1,32,128] bf16 TILE."""
    assert qpost_ok(qlin, C, S, R)
    mesh = qlin.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, 1, 32, 128]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh,
        ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    nt = R * 4
    cores = allc[:nt]
    cset = _merged_set(cores)
    bf = ttnn.bfloat16
    X, RC, SC, XS, O, R2, RI, PP, MM = range(9)
    cbs = [
        _cb(cset, X, 1, 2048, bf),
        _cb(cset, RC, 1, 2048, bf),
        _cb(cset, SC, 1, 256, bf),
        _cb(cset, XS, 1, 4096, bf),
        _cb(cset, O, 1, 2048, bf),
        _cb(cset, R2, 1, 2048, bf),
        _cb(cset, RI, 1, 2048, bf),
        _cb(cset, PP, 1, 2048, bf),
    ]
    rt = (
        ttnn.RuntimeArgs()
    )  # no per-core runtime arguments: the tile index is the row-major index of the core (ol_core_idx)
    reader = _kern(
        f"{KDIR}/qpost{_v2()}_reader.cpp",
        cset,
        [X, RC, (int(qlin.memory_config().buffer_type == ttnn.BufferType.L1) if _v2() else SC), XS, R2, RI]
        + _acc(qlin)
        + _acc(C)
        + _acc(S),
        rt,
        ttnn.ReaderConfigDescriptor(),
        [qlin.buffer_address(), C.buffer_address(), S.buffer_address(), int(grid.x)],
    )
    writer = _kern(
        f"{KDIR}/qpost_writer.cpp",
        cset,
        [O] + _acc(out),
        rt,
        ttnn.WriterConfigDescriptor(),
        [out.buffer_address(), int(grid.x)],
    )
    compute = _kern(
        f"{KDIR}/qpost{_v2()}_compute.cpp",
        cset,
        [X, RC, O, 1, R2, RI, MM, PP],
        ttnn.RuntimeArgs(),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x0A1,
        os.environ.get("DSV41_OL_ZONES", "0"),
        _v2(),
        R,
        tuple(_acc(qlin)),
        tuple(_acc(C)),
        tuple(_acc(S)),
        tuple(_acc(out)),
    )
    ttnn.generic_op([qlin, C, S, out], prog)
    return out
