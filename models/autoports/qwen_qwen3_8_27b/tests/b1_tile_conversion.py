# SPDX-License-Identifier: Apache-2.0
"""Experimental exact BF16 B1 tile conversion with bounded per-core staging."""
from pathlib import Path

import ttnn


def convert(x, height, memory_config):
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError("B1 tile conversion requires BF16 tiles")
    if list(x.shape)[:-1] != [1, 1, 1] or x.shape[-1] % 32:
        raise ValueError("B1 tile conversion requires [1,1,1,32*k]")
    source_height = x.tile.tile_shape[0]
    if source_height not in (16, 32) or height not in (16, 32) or height == source_height:
        raise ValueError("Only standard32/tiny16 conversion is supported")
    spec = ttnn.TensorSpec(
        ttnn.Shape(list(x.shape)),
        x.dtype,
        x.layout,
        memory_config.memory_layout,
        memory_config.shard_spec,
        memory_config.buffer_type,
        ttnn.Tile([height, 32]),
    )
    device = x.device()
    out = ttnn.allocate_tensor_on_device(spec, device)
    count = min(40, x.shape[-1] // 32)
    cores = ttnn.num_cores_to_corerangeset(count, device.compute_with_storage_grid_size(), row_wise=True)
    ct = [source_height * 64, height * 64, x.shape[-1] // 32, count]
    for tensor in (x, out):
        ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    grid = device.compute_with_storage_grid_size()
    for index in range(count):
        rt[index % grid.x][index // grid.x] = [x.buffer_address(), out.buffer_address(), index]
    staging = ttnn.CBDescriptor(
        total_size=2048,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("kernels") / "b1_tile_conversion.cpp"),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=[staging],
        semaphores=[],
    )
    ttnn.generic_op([x, out], descriptor)
    return out
