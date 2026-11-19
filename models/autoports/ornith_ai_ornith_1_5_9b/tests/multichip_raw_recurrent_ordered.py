# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token chunk-order experiment with unchanged raw pre-normalized inputs.

Normalization, exp(g), padding, head ordering, and state aliasing match the raw
helper. Only recurrence preparation/accumulation follows chunk GDN's one-real-
token specialization; compute.cpp documents the explicit FP32 pack boundaries.
"""
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels/raw_recurrent_ordered"


def raw_recurrent_ordered(q, k, v, decay, beta, state, wave=104):
    heads = state.shape[0]
    device = state.device()
    assert all(x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT for x in (q, k, v, decay, beta, state))
    assert tuple(state.shape) == (heads, 128, 128) and 1 <= wave <= 110
    assert heads >= 1
    assert all(tuple(x.shape) == (heads, 1, 128) for x in (q, k, v))
    assert all(tuple(x.shape) == (heads, 1, 1) for x in (decay, beta))
    assert all(not x.is_sharded() for x in (q, k, v, decay, beta, state))
    q = ttnn.reshape(q, q.shape, pad_value=0.0)
    k = ttnn.reshape(k, k.shape, pad_value=0.0)
    v = ttnn.reshape(v, v.shape, pad_value=0.0)
    out = ttnn.empty(
        (heads, 1, 128),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for start in range(0, heads, wave):
        count = min(wave, heads - start)
        cores = [ttnn.CoreCoord(h // 10, h % 10) for h in range(count)]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        # Explicit one-token prep and shared chunk scan CB capacities.
        sizes = {
            0: 4,
            1: 4,
            2: 4,
            3: 1,
            4: 1,
            5: 16,
            6: 4,
            7: 16,
            8: 4,
            9: 4,
            10: 4,
            11: 4,
            12: 1,
            13: 1,
            14: 1,
            16: 4,
            17: 4,
            18: 4,
            19: 4,
            20: 1,
        }
        cbs = [
            ttnn.CBDescriptor(
                total_size=n * 4096,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
            )
            for i, n in sizes.items()
        ]
        rct = [4, 4, 0]
        for x in (q, k, v, decay, beta, state, state):
            rct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        wct = (
            [4, 4, 0]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(state).get_compile_time_args()
        )
        rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            h = start + i
            rr[c.x][c.y] = [
                h,
                1,
                q.buffer_address(),
                k.buffer_address(),
                v.buffer_address(),
                decay.buffer_address(),
                beta.buffer_address(),
                state.buffer_address(),
                0,
                0,
                heads,
            ]
            wr[c.x][c.y] = [h, 1, out.buffer_address(), state.buffer_address(), heads]
            cr[c.x][c.y] = [1]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader.cpp"),
                core_ranges=grid,
                compile_time_args=rct,
                runtime_args=rr,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer.cpp"),
                core_ranges=grid,
                compile_time_args=wct,
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute.cpp"),
                core_ranges=grid,
                compile_time_args=[4, 4, 0],
                runtime_args=cr,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        ttnn.generic_op(
            [q, k, v, decay, beta, state, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
        )
    return out
