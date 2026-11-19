# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP32 four-tap convolution and in-place tiled history advancement."""
import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels/fast_zero_conv_gates"


def conv_gates_precise(raw, history, taps, bias, decay):
    assert raw.dtype == history.dtype == ttnn.bfloat16
    assert tuple(raw.shape)[:2] == (1, 1) and raw.shape[-1] == 3136
    assert tuple(history.shape) == (3, 1, raw.shape[2], 2048)
    assert 1 <= raw.shape[2] <= 32 and len(taps) == 4
    assert all(t.dtype == ttnn.float32 and math.prod(t.shape) == 8 for t in (bias, decay))
    for tensor in (raw, history, *taps, bias, decay):
        assert tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert all(t.dtype == ttnn.float32 for t in taps)
    out = ttnn.empty(
        (1, 1, raw.shape[2], 2048),
        device=raw.device(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    gates = [
        ttnn.empty(
            (1, 1, raw.shape[2], 8),
            device=raw.device(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for dtype in (ttnn.float32, ttnn.bfloat16)
    ]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    cbs = []
    for idx in (*range(14), 16, 17, 18):
        dtype = ttnn.float32 if idx in (4, 5, 6, 7, 10, 11, 12, 13, 17) else ttnn.bfloat16
        size = 4096 if dtype == ttnn.float32 else 2048
        cbs.append(
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)],
            )
        )
    rct = []
    for tensor in (raw, history, *taps, bias, decay):
        rct += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    wct = (
        ttnn.TensorAccessorArgs(out).get_compile_time_args() + ttnn.TensorAccessorArgs(history).get_compile_time_args()
    )
    wct += (
        ttnn.TensorAccessorArgs(gates[0]).get_compile_time_args()
        + ttnn.TensorAccessorArgs(gates[1]).get_compile_time_args()
    )
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(64):
        x, y = i % 8, i // 8
        rr[x][y] = [
            i,
            raw.buffer_address(),
            history.buffer_address(),
            *[t.buffer_address() for t in taps],
            bias.buffer_address(),
            decay.buffer_address(),
        ]
        wr[x][y] = [
            i,
            out.buffer_address(),
            history.buffer_address(),
            gates[0].buffer_address(),
            gates[1].buffer_address(),
        ]
        cr[x][y] = [i]
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
    for index in (10, 11, 12, 13):
        unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    # This checkout binds unpack modes as a writable field, not a constructor keyword.
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
            runtime_args=cr,
            config=compute_config,
        ),
    ]
    ttnn.generic_op(
        [raw, history, *taps, bias, decay, out, *gates], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return out, *gates


# Process-local probe compatibility.
conv_gates = conv_gates_precise
