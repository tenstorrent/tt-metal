# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flat-input preparation plus the passing one-token chunk-order recurrence.

FP32/HiFi4 arithmetic and the aligned-64-byte DRAM reader are preserved. CB15
holds ordered diagonal decay, freeing CB20 for the flat BF16 Q input; CB14 is
its own FP32 identity, separate from the BF16 normalization scaler in CB27.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels/flat_ordered_conv_gates"
MATH_FIDELITY = ttnn.MathFidelity.HiFi4


def flat_ordered_conv_gates(conv, g, beta, state, gate, gamma, epsilon, wave=104, rows=None):
    """`rows` (decode buckets): process only pool rows 0..rows-1 (heads 0..8*rows-1) of a larger ordered state pool,
    in place; the other rows are neither read nor written."""
    heads = state.shape[0] if rows is None else 8 * rows
    assert gate.dtype == gamma.dtype == ttnn.bfloat16
    assert tuple(gate.shape) == (1, 1, heads // 8, 3136) and gamma.shape[-1] == 128
    assert heads % 8 == 0 and 1 <= heads // 8 <= 32
    assert state.shape[0] % 8 == 0 and heads <= state.shape[0] and tuple(state.shape)[1:] == (128, 128)
    assert 1 <= wave <= 110
    for tensor in (conv, g, beta, state, gate, gamma):
        assert tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert conv.dtype == beta.dtype == ttnn.bfloat16 and g.dtype == state.dtype == ttnn.float32
    assert tuple(conv.shape) == (1, 1, heads // 8, 2048) and tuple(g.shape) == tuple(beta.shape) == (
        1,
        1,
        heads // 8,
        8,
    )
    out = ttnn.empty(
        (1, 1, heads // 8, 1024),
        device=state.device(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    for start in (0,):
        count = min(wave, heads - start)
        cores = [ttnn.CoreCoord(h // 10, h % 10) for h in range(count)]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
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
            15: 1,
            16: 4,
            17: 4,
            18: 4,
            19: 4,
            20: 4,
            21: 4,
            22: 4,
            23: 1,
            24: 1,
            25: 4,
            26: 1,
            27: 1,
            28: 1,
            29: 4,
            30: 4,
            31: 4,
            32: 16,  # FP32 state increment k_t @ v_new
            33: 16,  # FP32 copy of the state, unpacked straight to DST
            34: 1,  # FP32 exp(g) column, unpacked straight to DST
        }
        cbs = []
        for i, n in sizes.items():
            dtype = ttnn.bfloat16 if i in (20, 21, 22, 24, 27, 29, 30, 31) else ttnn.float32
            size = 2048 if dtype == ttnn.bfloat16 else 4096
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=n * size,
                    core_ranges=grid,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size)],
                )
            )
        rct = []
        for x in (conv, g, beta, state, gate, gamma):
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
                conv.buffer_address(),
                g.buffer_address(),
                beta.buffer_address(),
                state.buffer_address(),
                gate.buffer_address(),
                gamma.buffer_address(),
                count,
                heads,
            ]
            wr[c.x][c.y] = [h, 1, out.buffer_address(), state.buffer_address(), heads, count]
            cr[c.x][c.y] = [(heads - h + count - 1) // count]
        compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=MATH_FIDELITY, fp32_dest_acc_en=True, math_approx_mode=False
        )
        # The scan decays the recurrent state on the SFPU in FP32: g (CB23), the increment (CB32), the state copy
        # (CB33) and exp(g) (CB34) are unpacked straight to DST. The default srcA path holds at most TF32 and
        # truncates; CB3/CB5 keep that path because FPU (matmul / broadcast) consumers read them.
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
        for index in (23, 32, 33, 34):
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes
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
                compile_time_args=[4, 4, 0, struct.unpack("<I", struct.pack("<f", epsilon))[0]],
                runtime_args=cr,
                config=compute_config,
            ),
        ]
        ttnn.generic_op(
            [conv, g, beta, state, gate, gamma, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
        )
    return out
