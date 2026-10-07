# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT top-K router selection (``ttnn.generic_op``), ported from the DSV4.1 router_select (kernel copied unchanged): one core per token row, exact fp32 top-K
(lowest index wins ties) + normalised bf16 weights + uint16 indices in one program. Replaces ttnn.topk + typecasts + gather + normalise (~0.14 ms at T=64)."""
import os
import struct

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_kernels")


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


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


def router_select(rank, score, k, eps, scale):
    """rank, score: fp32 TILE [1,1,T,E] (E % 32 == 0, T <= 32; rank = score + bias).
    Returns (weights bf16 RM [T,1,1,k], indices uint16 RM [T,1,1,k]) in DRAM."""
    _, _, T, E = (int(v) for v in rank.shape)
    assert T <= 32 and E % 32 == 0 and rank.dtype == ttnn.float32 and score.dtype == ttnn.float32
    mesh = rank.device()
    mk = lambda dt: ttnn.allocate_tensor_on_device(
        ttnn.Shape([T, 1, 1, k]), dt, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    w, idx = mk(ttnn.bfloat16), mk(ttnn.uint16)
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:T]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for t, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [t]
    f32 = ttnn.float32
    cbs = [_cb(core_set, 0, 1, E * 4, f32), _cb(core_set, 1, 1, 6 * 64, f32), _cb(core_set, 2, 1, 128, f32)]
    ct = [0, 1, 2, E // 32, k, _bits(eps), _bits(scale)] + _acc(rank) + _acc(score) + _acc(w) + _acc(idx)
    kern = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/router_select.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[rank.buffer_address(), score.buffer_address(), w.buffer_address(), idx.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5B1,
        T,
        E,
        k,
        _bits(eps),
        _bits(scale),
        tuple(_acc(rank)),
        tuple(_acc(score)),
        tuple(_acc(w)),
        tuple(_acc(idx)),
    )
    ttnn.generic_op([rank, score, w, idx], prog)
    return w, idx
