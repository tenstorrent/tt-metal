# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct aligned row writes into persistent tiled convolution history."""

from pathlib import Path

import ttnn


def history_row_write(value, pool, row):
    assert value.dtype == pool.dtype == ttnn.bfloat16
    assert value.layout == ttnn.ROW_MAJOR_LAYOUT and pool.layout == ttnn.TILE_LAYOUT
    assert tuple(value.shape) == (1, 1, 3, 2048) and tuple(pool.shape)[:2] == (3, 1) and pool.shape[-1] == 2048
    assert 0 <= row < pool.shape[2] <= 32
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))])
    runtime = ttnn.RuntimeArgs()
    for i in range(16):
        runtime[i % 8][i // 8] = [value.buffer_address(), pool.buffer_address(), row, i * 128]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/history_row_write/write.cpp"),
        core_ranges=grid,
        compile_time_args=ttnn.TensorAccessorArgs(value).get_compile_time_args()
        + ttnn.TensorAccessorArgs(pool).get_compile_time_args(),
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=256)],
    )
    ttnn.generic_op([value, pool], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))
