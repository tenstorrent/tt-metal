# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode attention prologue (1..32 rows: batch 1, the DFlash verify, batch-32 decode) as one generic_op (Laguna;
kernels/ap1_*.cpp).

Replaces the head split of the fused qkv matmul output, the per-head q / k RMSNorms, the HF rotate_half RoPE and
the layout moves between them (~16 ops at batch 1, ~11 at 6 rows). For each token row b, core (b, 0) builds q and
core (b, 1) builds k and copies v: each gathers row b of the qkv output into a head-major [32 (heads), 128] block,
then RMSNorm + RoPE run on the tile engine."""

import os
import struct
from pathlib import Path

import torch
import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def _tile_bytes(dt):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}[dt]


def _cb(grid, i, fmt, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
    )


def _core_ranges(gx, start, count):
    """CoreRangeSet of the row-major core indices [start, start + count)."""
    ranges, i = set(), start
    while i < start + count:
        y, x = divmod(i, gx)
        n = min(gx - x, start + count - i)
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + n - 1, y)))
        i += n
    return ttnn.CoreRangeSet(ranges)


def reduce_scaler(device, head_dim, mesh_mapper=None):
    """bf16 tile of 1 / head_dim (the row-reduce scaler: the reduced sum is the mean)."""
    return ttnn.from_torch(
        torch.full((1, 1, 32, 32), 1.0 / head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        mesh_mapper=mesh_mapper,
    )


def attn_prologue1(qkv, cos, sin, q_norm, k_norm, scaler, nq, nkv, rd, eps, k_mem, v_mem, rows=1, row0=0):
    """qkv: [.., R, W] bf16 TILE (this call's rows: row0 .. row0 + rows - 1, row0 + rows <= R; [q heads | k heads |
    v heads | ...], head_dim 128; the batched verify passes all its rows once, one 32-row chunk per call); cos / sin:
    [1, rows, 1, rd] bf16 TILE (one row per token); q_norm / k_norm: [1, 1, 1, 128] TILE weight rows. Returns q
    [1, rows, nq, 128] (DRAM), k [1, rows, nkv, 128] in k_mem and v in v_mem (the KV-update height shards). More than
    32 rows (the batched DFlash verify: 64 / 96, one op instead of one per 32-row chunk): lists of one q / k / v per
    32-row chunk (k_mem / v_mem: a chunk's layout); each core then handles several consecutive rows."""
    device = qkv.device()
    hd = 128
    assert qkv.dtype == ttnn.bfloat16 and qkv.padded_shape[-1] >= (nq + 2 * nkv) * hd and nq <= 32 and nkv <= 32
    assert rd in (64, 128), rd
    assert 1 <= rows <= 96 and (rows <= 32 or rows % 32 == 0), rows
    assert row0 + rows <= qkv.padded_shape[-2], (rows, row0, qkv.padded_shape)
    wt_qkv = int(qkv.padded_shape[-1]) // 32
    n_chunks = -(-rows // 32)
    cr = min(rows, 32)  # rows per chunk tensor

    def alloc(heads, mem):
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, cr, heads, hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                              mem)  # fmt: skip

    qs = [alloc(nq, ttnn.DRAM_MEMORY_CONFIG) for _ in range(n_chunks)]
    ks = [alloc(nkv, k_mem) for _ in range(n_chunks)]
    vs = [alloc(nkv, v_mem) for _ in range(n_chunks)]
    q, k, v = qs[0], ks[0], vs[0]
    rd_tiles = rd // 32
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    w_page, cs_page = _tile_bytes(q_norm.dtype), _tile_bytes(cos.dtype)
    kernels, cbs = [], []
    # q heads over QP cores per row (core y = part, TT_LAGUNA_AP1_QPARTS; even head counts per part), k on y = QP
    gs = device.compute_with_storage_grid_size()
    # q heads over qp cores per row (TT_LAGUNA_AP1_QPARTS; even head counts per part), then one k core per row; the
    # cores are consecutive row-major indices (q: [0, rows * qp), k: the next rows)
    qp = max(1, int(os.environ.get("TT_LAGUNA_AP1_QPARTS", "3" if rows <= 8 else "2")))
    hp = -(-nq // qp)
    hp += hp % 2
    qp = -(-nq // hp)
    # rows per core: the fewest (dividing rows) whose cores fit the grid
    rpc = next(r for r in range(1, rows + 1) if rows % r == 0 and (rows // r) * (qp + 1) <= gs.x * gs.y)
    groups = rows // rpc
    for role, (w, nh, off, out) in enumerate(((q_norm, nq, 0, q), (k_norm, nkv, nq * 4, k))):
        base, parts = (0, qp) if role == 0 else (groups * qp, 1)
        grid = _core_ranges(gs.x, base, groups * parts)
        per_part = hp if role == 0 else nh
        acc = []
        for t in (qkv, w, cos, sin, scaler):
            acc.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "ap1_reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[role, per_part, off, (nq + nkv) * 4, rd_tiles, w_page, cs_page, nq, gs.x, base, parts,
                               row0, wt_qkv, rpc]
            + acc,
            common_runtime_args=[qkv.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                                 scaler.buffer_address(), w.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        ))
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "ap1_writer.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[role, hp, nq, qp if role == 0 else 1, gs.x, base, rpc]
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(v).get_compile_time_args()),
            # per chunk: its output (q or k), then its v
            common_runtime_args=[a for c in range(n_chunks)
                                 for a in ((qs if role == 0 else ks)[c].buffer_address(), vs[c].buffer_address())],  # fmt: skip
            config=ttnn.WriterConfigDescriptor(),
        ))
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.fp32_dest_acc_en = True
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "ap1_compute.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[rd_tiles, eps_bits, rpc],
            config=cfg,
        ))
        db = 2 if rpc > 1 else 1  # several rows per core: double-buffer the per-row inputs / outputs
        cbs += [
            _cb(grid, 0, ttnn.bfloat16, 2048, 4 * db),
            _cb(grid, 1, w.dtype, w_page, 4),
            _cb(grid, 2, cos.dtype, cs_page, rd_tiles * db),
            _cb(grid, 3, sin.dtype, cs_page, rd_tiles * db),
            _cb(grid, 4, ttnn.bfloat16, 2048, 1),
            _cb(grid, 5, ttnn.bfloat16, 2048, 4 * db),
            _cb(grid, 16, ttnn.bfloat16, 2048, 4 * db),
            _cb(grid, 24, ttnn.float32, 4096, 4),
            _cb(grid, 25, ttnn.float32, 4096, 1),
            _cb(grid, 26, ttnn.float32, 4096, 4),
            _cb(grid, 27, ttnn.bfloat16, 2048, 4),
        ]
    ttnn.generic_op(
        [qkv, q_norm, k_norm, cos, sin, scaler] + qs + ks + vs,
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    return (q, k, v) if n_chunks == 1 else (qs, ks, vs)
