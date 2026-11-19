# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP32 four-tap convolution and in-place tiled history advancement."""
from pathlib import Path
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels/tiled_conv"


def packed_tiled_conv(raw, history, taps):
    assert raw.dtype == history.dtype == ttnn.bfloat16
    assert tuple(raw.shape)[:2] == (1, 1) and raw.shape[-1] == 3136
    assert tuple(history.shape) == (3, 1, raw.shape[2], 2048)
    assert 1 <= raw.shape[2] <= 32 and len(taps) == 4
    for tensor in (raw, history, *taps):
        assert tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert all(t.dtype == ttnn.float32 for t in taps)
    out = ttnn.empty((1,1,raw.shape[2],2048), device=raw.device(), dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    cbs = []
    for idx in (*range(8), 16):
        dtype = ttnn.float32 if 4 <= idx < 8 else ttnn.bfloat16
        size = 4096 if dtype == ttnn.float32 else 2048
        cbs.append(ttnn.CBDescriptor(total_size=size, core_ranges=grid,
                   format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)]))
    rct = []
    for tensor in (raw, history, *taps):
        rct += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    wct = ttnn.TensorAccessorArgs(out).get_compile_time_args() + ttnn.TensorAccessorArgs(history).get_compile_time_args()
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(64):
        x, y = i % 8, i // 8
        rr[x][y] = [i, raw.buffer_address(), history.buffer_address(), *[t.buffer_address() for t in taps]]
        wr[x][y] = [i, out.buffer_address(), history.buffer_address()]
        cr[x][y] = [0]
    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), core_ranges=grid,
                             compile_time_args=rct, runtime_args=rr, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "writer.cpp"), core_ranges=grid,
                             compile_time_args=wct, runtime_args=wr, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "compute.cpp"), core_ranges=grid,
                             runtime_args=cr, config=ttnn.ComputeConfigDescriptor(
                                 math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
    ]
    ttnn.generic_op([raw, history, *taps, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out
