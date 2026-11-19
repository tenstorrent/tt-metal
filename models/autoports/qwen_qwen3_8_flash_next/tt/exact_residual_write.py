# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact local2560 residual writer for one stream per mesh rank.

One complete BF16 stream per rank. Preserve FP32 product -> BF16 update ->
FP32 residual add -> BF16 output, using the inherited native SFPU implementation.
"""

from pathlib import Path

import ttnn

_KERNELS = Path(__file__).parent / "kernels"


def exact_residual_write(residual, block, injection):
    if not all(isinstance(t, ttnn.Tensor) for t in (residual, block, injection)):
        raise ValueError("Expected device TTNN tensors")
    count = residual.shape[-2]
    device = residual.device()
    if tuple(device.shape) != (1, 4) or not 1 <= count <= 32:
        raise ValueError("Expected local2560 TP4 stream and physical rows1..32")
    padded_count = ((count + 31) // 32) * 32
    for value, width in ((residual, 2560), (block, 2560), (injection, 1)):
        tile = value.get_tile()
        if (
            not ttnn.is_tensor_storage_on_device(value)
            or not value.is_allocated()
            or value.device() != device
            or tuple(value.shape) != (1, 1, count, width)
            or tuple(value.padded_shape) != (1, 1, padded_count, ((width + 31) // 32) * 32)
            or value.dtype != ttnn.bfloat16
            or value.layout != ttnn.TILE_LAYOUT
            or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tuple(tile.tile_shape) != (32, 32)
            or tile.transpose_within_face
            or tile.transpose_of_faces
        ):
            raise ValueError("Expected allocated local BF16 standard TILE DRAM tensors with matching rows/device")
    output = ttnn.empty_like(residual)
    grid = device.compute_with_storage_grid_size()
    total = padded_count // 32 * 80
    used = min(total, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(used)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        start, end = total * i // used, total * (i + 1) // used
        reader[core.x][core.y] = [
            block.buffer_address(),
            injection.buffer_address(),
            residual.buffer_address(),
            start,
            end - start,
        ]
        compute[core.x][core.y] = [end - start]
        writer[core.x][core.y] = [output.buffer_address(), start, end - start]
    read_ct = []
    for value in (block, injection, residual):
        read_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    specs = [
        ("reader", read_ct, reader, ttnn.ReaderConfigDescriptor()),
        (
            "compute",
            [],
            compute,
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
                dst_full_sync_en=False,
            ),
        ),
        ("writer", ttnn.TensorAccessorArgs(output).get_compile_time_args(), writer, ttnn.WriterConfigDescriptor()),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / f"exact_residual_write_{role}.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for role, ct, rt, config in specs
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op(
        [block, injection, residual, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output
