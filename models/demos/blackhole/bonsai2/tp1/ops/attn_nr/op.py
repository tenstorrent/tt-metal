# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attention decode q/k RMSNorm + RoPE as one device op (QWEN36_1C_ATTN_NR), via ttnn.generic_op.

Replaces, per full-attention decode layer (with ATTN_LEAN + ATTN_QGVIEW): rms_norm(q) on [Hq,1,D], the permute to
[1,1,Hq,D], k's reshape to [1,1,Hk,D], rms_norm(k), and the two 4-op RoPE sequences (12 ops, ~54 us/layer on the
ternary stack, fconv2 profile) with one op:

    q      [Hq, 1, D]  bf16 TILE (QGVIEW's q slice: one head per tile-row)
    k      [1, 1, Hk*D] bf16 TILE (the fused-qkv k slice)
    wq, wk the q/k norm weights (bf16 TILE, row 0 = D values, +1 pre-offset)
    cos, sin [1, 1, rotary_dim] bf16 TILE (only the Rt rotary tiles are read)
  ->
    q_out [1, 1, Hq, D], k_out [1, 1, Hk, D] bf16, into caller-owned persistent zeroed buffers (alloc_outputs).

One core per head (Hq + Hk cores, one rectangle). RMSNorm and RoPE in fp32 DST, so not bit-exact with the bf16 chain;
test_attn_nr.py requires it at least as close to float64, with magnitude and shape checks.
"""

from __future__ import annotations

import struct

import torch
import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tp1/ops/attn_nr"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _cb(index, pages, dtype, cores):
    page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def alloc_outputs(device, hq, hk, d, k_mc=ttnn.L1_MEMORY_CONFIG):
    """Persistent zeroed q_out (DRAM, as paged SDPA decode wants it) and k_out (k_mc: the model passes the
    height-sharded L1 config paged_update_cache takes, so k needs no to_memory_config)."""

    def z(h, mc):
        return ttnn.from_torch(torch.zeros(1, 1, h, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                               memory_config=mc)

    return z(hq, ttnn.DRAM_MEMORY_CONFIG), z(hk, k_mc)


def attn_nr(q, k, wq, wk, cos, sin, q_out, k_out, eps, rotary_dim):
    hq, D = q.shape[0], q.shape[2]
    hk = k_out.shape[2]
    Dt, Rt = D // 32, rotary_dim // 32
    assert Rt % 2 == 0 and k.shape[-1] == hk * D, (rotary_dim, k.shape)
    n = hq + hk
    gw = 7 if n % 7 == 0 else n
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gw - 1, n // gw - 1))])
    cbs = [
        _cb(0, Dt, ttnn.bfloat16, cores),  # x
        _cb(1, Dt, ttnn.bfloat16, cores),  # norm weight
        _cb(2, 2 * Rt, ttnn.bfloat16, cores),  # cos tiles, then sin tiles
        _cb(3, 1, ttnn.float32, cores),  # sum of squares
        _cb(4, 1, ttnn.float32, cores),  # rinv
        _cb(5, Dt, ttnn.float32, cores),  # normalized
        _cb(6, Dt, ttnn.bfloat16, cores),  # out
        _cb(7, max(1, -(-(Dt * 2 * 128) // 2048)), ttnn.bfloat16, cores),  # writer staging, 128 B per run
    ]
    roles = [((c % gw, c // gw), (0, c) if c < hq else (1, c - hq)) for c in range(n)]
    addrs = [q.buffer_address(), k.buffer_address(), wq.buffer_address(), wk.buffer_address(),
             cos.buffer_address(), sin.buffer_address()]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/reader_attn_nr.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[Dt, Rt, *_acc(q), *_acc(k), *_acc(wq), *_acc(wk), *_acc(cos), *_acc(sin)],
        runtime_args=[(c, [*addrs, r, h]) for c, (r, h) in roles],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/compute_attn_nr.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[Dt, Rt],
        runtime_args=[(c, [_bits(1.0 / D), _bits(eps)]) for c, _ in roles],
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/writer_attn_nr.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[Dt, *_acc(q_out), *_acc(k_out)],
        runtime_args=[(c, [q_out.buffer_address(), k_out.buffer_address(), r, h]) for c, (r, h) in roles],
        config=ttnn.WriterConfigDescriptor(),
    )
    ttnn.generic_op([q, k, wq, wk, cos, sin, q_out, k_out],
                    ttnn.ProgramDescriptor(kernels=[reader, compute, writer], cbs=cbs))
    return q_out, k_out
