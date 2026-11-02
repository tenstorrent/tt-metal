# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused head-major GDN decode conv (QWEN36_1C_GDN_FCONV), one device op via ttnn.generic_op.

Replaces, per GDN decode layer: the qkv row -> [1, 80, 128] reshape (15.6 us), the K state copies, mul + K-1 mac +
SiLU, and GDN_HMI's fp32 typecast, permute and three q/k/v slices (2.8 ms/step together on the ternary stack,
22:08 re-budget).

Contract (one user, one token, D = 128):
    qkv    [1, 1, C]     bf16 TILE, the in-projection's qkv row (C = (2*NK + NV) * 128)
    state  K x [1, R, 128] bf16 TILE, CONV_HM's shift register (R = 2*NK + NV heads), updated IN PLACE
    taps   K x [1, R, 128] bf16 TILE, CONV_HM's head-major taps
  ->
    q [NK, 1, 128], k [NK, 1, 128], v [NV, 1, 128]  fp32 TILE, one head per tile-row (what the recurrence takes
    with head_major_input), written into caller-owned persistent buffers whose padding rows stay zero.

One core per state tile (r_t, c_t): 12 cores as one 4 x 3 rectangle (a scattered set costs dispatch per range in a
trace; see GDN_RECT). conv = sum_j tap[j] * state_new[j] runs in fp32 DST, so it is NOT bit-exact with the bf16
ttnn chain (which rounds after every tap); it is closer to float64 (test_conv_hm_fused.py).
"""

from __future__ import annotations

import torch
import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tp1/ops/conv_hm"
D = 128


def _cb(index, pages, dtype, cores):
    page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def alloc_outputs(device, nk, nv, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Persistent zeroed q/k/v buffers (allocate once, before any trace)."""

    def z(n):
        return ttnn.from_torch(
            torch.zeros(n, 1, D), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    return z(nk), z(nk), z(nv)


def conv_hm(qkv, state, taps, q, k, v):
    """See the module docstring. Updates `state` in place and writes q, k, v; returns (q, k, v)."""
    K = len(state)
    nk, nv = q.shape[0], v.shape[0]
    R = 2 * nk + nv
    assert tuple(state[0].shape) == (1, R, D) and tuple(taps[0].shape) == (1, R, D), (state[0].shape, taps[0].shape)
    assert qkv.shape[-1] == R * D, qkv.shape
    rt_n = -(-R // 32)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(D // 32 - 1, rt_n - 1))])

    cbs = [
        _cb(0, 1, ttnn.bfloat16, cores),  # new row, for compute
        _cb(1, K - 1, ttnn.bfloat16, cores),  # old state rows 1..K-1, for compute
        _cb(2, K, ttnn.bfloat16, cores),  # taps
        _cb(3, 1, ttnn.float32, cores),  # conv + silu, fp32
        _cb(4, K, ttnn.bfloat16, cores),  # old rows 1..K-1 + new row, for the writer
        _cb(5, 2, ttnn.bfloat16, cores),  # reader scratch: 64 B-aligned slots for the 2 x 32 B runs per head
    ]
    rd_rt, wr_rt = [], []
    for r_t in range(rt_n):
        rows = min(32, R - 32 * r_t)
        for c_t in range(D // 32):
            core = (c_t, r_t)
            rd_rt.append(
                (core, [qkv.buffer_address(), *[s.buffer_address() for s in state], *[t.buffer_address() for t in taps],
                        r_t, c_t, rows])
            )
            wr_rt.append(
                (core, [*[s.buffer_address() for s in state], q.buffer_address(), k.buffer_address(), v.buffer_address(),
                        r_t, c_t, rows])
            )
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/reader_conv_hm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[K, *_acc(qkv), *_acc(state[0]), *_acc(taps[0])],
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/compute_conv_hm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[K],
        runtime_args=[(core, []) for core, _ in rd_rt],
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/writer_conv_hm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[K, nk, *_acc(state[0]), *_acc(q)],
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    ttnn.generic_op([qkv, *state, *taps, q, k, v], ttnn.ProgramDescriptor(kernels=[reader, compute, writer], cbs=cbs))
    return q, k, v
