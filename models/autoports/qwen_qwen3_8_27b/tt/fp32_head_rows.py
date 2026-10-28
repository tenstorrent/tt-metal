# SPDX-License-Identifier: Apache-2.0
"""B1 exact head-row repack; preserves FP32 normalization policy."""

import math
from pathlib import Path

import ttnn


def repack(q, k, v):
    for tensor, width in zip((q, k, v), (512, 512, 1536)):
        if (
            tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.TILE_LAYOUT
            or math.prod(list(tensor.shape)[:-2]) != 1
            or tensor.shape[-1] != width
            or tensor.padded_shape[-2] != 32
            or list(tensor.tile.tile_shape) != [32, 32]
        ):
            raise ValueError("Head repack requires one BF16 32-row tile strip, widths512/512/1536")
    device = q.device()
    outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, heads, 128]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for heads in (4, 4, 12)
    ]
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(12, grid, row_wise=True)
    ct = []
    for tensor in (q, k, v, *outputs):
        ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for worker in range(12):
        rt[worker % grid.x][worker // grid.x] = [tensor.buffer_address() for tensor in (q, k, v, *outputs)] + [worker]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=page)],
        )
        for index, size, page in ((0, 12288, 1024), (1, 4096, 4096))
    ]
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("kernels") / "repack_fp32_head_rows.cpp"),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=cbs,
        semaphores=[],
    )
    ttnn.generic_op([q, k, v, *outputs], descriptor)
    return outputs
