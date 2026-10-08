# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT top-K router selection (``ttnn.generic_op``): one core per token row, exact fp32 top-K + normalised bf16 weights.

Replaces ``ttnn.topk`` (63-68 us on one core) and the one-hot / select / normalise / layout chain (~40 us) of the fp32
exact router with one program of ~T parallel scalar cores."""

import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def router_select(rank, score, k, eps, scale):
    """rank, score: fp32 TILE [1,1,T,E] (E % 32 == 0, T % 32 == 0 or T <= 32; rank = score + bias).
    Returns (weights bf16 RM [T,1,1,k], indices uint16 RM [T,1,1,k]) in DRAM."""
    _, _, T, E = (int(v) for v in rank.shape)
    assert (T <= 32 or T % 32 == 0) and E % 32 == 0 and rank.dtype == ttnn.float32 and score.dtype == ttnn.float32
    mesh = rank.device()
    mk = lambda dt: ttnn.allocate_tensor_on_device(
        ttnn.Shape([T, 1, 1, k]), dt, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    w, idx = mk(ttnn.bfloat16), mk(ttnn.uint16)
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][: min(T, grid.x * grid.y)]
    nc = len(cores)
    cnt = [
        len(range(i, T, nc)) for i in range(nc)
    ]  # row i, i + nc, ... of core i (T <= 32: one row per core, as before)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for t, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [t, nc, cnt[t]]
    f32 = ttnn.float32
    cbs = [_cb(core_set, 0, 1, E * 4, f32), _cb(core_set, 1, 1, 6 * 64, f32), _cb(core_set, 2, 1, 128, f32)]
    ct = [0, 1, 2, E // 32, k, _bits(eps), _bits(scale)] + _acc(rank) + _acc(score) + _acc(w) + _acc(idx)
    kern = _kernel(
        "router_select.cpp",
        core_set,
        ct,
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[rank.buffer_address(), score.buffer_address(), w.buffer_address(), idx.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5B2,
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
