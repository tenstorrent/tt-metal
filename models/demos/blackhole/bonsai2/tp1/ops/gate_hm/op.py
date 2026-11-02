# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN decode gated norm + SiLU(z) gate as one device op (QWEN36_1C_GDN_GATE), via ttnn.generic_op.

Replaces, per GDN decode layer, rms_norm(o, weight=w, epsilon=eps) on the recurrence's head-major o, the zero-copy
flatten, silu(z) and the multiply (three ops, ~0.6 ms/step on the ternary stack) with one op:

    o    [H, 1, D] fp32 TILE  (the recurrence's head_major_output)
    z    [1, 1, H*D] bf16 TILE (the output gate from the in-projection)
    w    [1, 1, D] bf16 TILE   (tw["norm_w"])
  ->
    out  [1, 1, H*D] bf16 or fp32 TILE, gated = o_h * rsqrt(mean(o_h^2) + eps) * w * silu(z_h), written into a caller-owned
         persistent zeroed buffer (row 0 only, so its padding rows stay zero).

One core per head, H cores as one rectangle (GDN_RECT's lesson). fp32 in DST throughout, so not bit-exact with the
bf16-output LN/silu/mul chain; it must be at least as close to float64 (test_gate_hm.py).
v2: the reader packs head h's Dt live rows (row 0 of Dt tiles) into rows 0..Dt-1 of ONE tile, so each SFPU op runs
once per head, not Dt times, and rinv is computed once (v1, a measured null: 18.8 us vs ~13 us for the chain).
"""

from __future__ import annotations

import struct

import torch
import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tp1/ops/gate_hm"


def _f32_bits(x: float) -> int:
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


def alloc_output(device, width, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG):
    """Persistent zeroed [1, 1, width] output (allocate once, before any trace)."""
    return ttnn.from_torch(
        torch.zeros(1, 1, width), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def gate_hm(o, z, w, out, eps=1e-6, grid_w=8, stage=3):
    H, _, D = o.shape
    Dt = D // 32
    assert z.shape[-1] == H * D and out.shape[-1] == H * D and w.shape[-1] == D, (o.shape, z.shape, w.shape)
    rows = -(-H // grid_w)
    assert H == rows * grid_w, f"{H} heads must fill a {grid_w}-wide rectangle"
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, rows - 1))])
    assert Dt <= 16, "packed rows must stay in faces 0/1"
    cbs = [
        _cb(0, Dt, ttnn.float32, cores),  # o, row 0 of tile t replicated in rows 0..Dt-1 (sum-of-squares source)
        _cb(1, 1, ttnn.bfloat16, cores),  # z, packed
        _cb(2, 1, ttnn.bfloat16, cores),  # w, packed
        _cb(4, 1, out.dtype, cores),  # gated, packed, in the output's dtype (bf16 for the ternary prep, fp32 dense)
        _cb(5, 1, ttnn.float32, cores),  # o, packed
        _cb(6, 1, ttnn.bfloat16, cores),  # reader staging (z/w 32 B runs at 64 B slots)
        _cb(7, 1, ttnn.bfloat16, cores),  # writer staging
    ]
    heads = [((h % grid_w, h // grid_w), h) for h in range(H)]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/reader_gate_hm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[Dt, *_acc(o), *_acc(z), *_acc(w)],
        runtime_args=[(c, [o.buffer_address(), z.buffer_address(), w.buffer_address(), h]) for c, h in heads],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/compute_gate_hm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[Dt, stage],  # stage < 3: debug output of an intermediate (test_gate_hm.py)
        runtime_args=[(c, [_f32_bits(1.0 / D), _f32_bits(eps)]) for c, _ in heads],
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/writer_gate_hm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[Dt, 2 if out.dtype == ttnn.bfloat16 else 4, *_acc(out)],
        runtime_args=[(c, [out.buffer_address(), h]) for c, h in heads],
        config=ttnn.WriterConfigDescriptor(),
    )
    ttnn.generic_op([o, z, w, out], ttnn.ProgramDescriptor(kernels=[reader, compute, writer], cbs=cbs))
    return out
