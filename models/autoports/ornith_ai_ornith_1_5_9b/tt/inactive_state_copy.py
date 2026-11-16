# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Copy only inactive rows of a native FP32 recurrent pool, preserving bits."""

from pathlib import Path

import ttnn


def inactive_state_copy(source, destination, active):
    if source.shape != destination.shape or tuple(source.shape)[1:] != (8, 128, 128):
        raise ValueError("Expected matching native [B,8,128,128] recurrent pools")
    if not 1 <= source.shape[0] <= 32:
        raise ValueError("Recurrent batch capacity must be in [1,32]")
    if source.dtype != ttnn.float32 or destination.dtype != ttnn.float32:
        raise ValueError("Inactive state copies preserve FP32 recurrence")
    if source.layout != ttnn.TILE_LAYOUT or destination.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Recurrent pools must be tiled")
    if source.memory_config() != ttnn.DRAM_MEMORY_CONFIG or destination.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("Recurrent pools must be DRAM interleaved")
    if tuple(active.shape) != (1, 32) or active.dtype != ttnn.int32 or active.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("Active rows require a persistent row-major int32[1,32] mask")
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    args = ttnn.RuntimeArgs()
    for core in range(64):
        args[core % 8][core // 8] = [
            source.buffer_address(),
            destination.buffer_address(),
            active.buffer_address(),
            source.shape[0],
            core,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/inactive_state_copy/copy.cpp"),
        core_ranges=grid,
        compile_time_args=ttnn.TensorAccessorArgs(source).get_compile_time_args()
        + ttnn.TensorAccessorArgs(destination).get_compile_time_args()
        + ttnn.TensorAccessorArgs(active).get_compile_time_args(),
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    data = ttnn.CBDescriptor(
        total_size=4096,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4096)],
    )
    mask = ttnn.CBDescriptor(
        total_size=128,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.int32, page_size=128)],
    )
    ttnn.generic_op(
        [source, destination, active], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[data, mask], semaphores=[])
    )
