# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact local reducer after the selected synchronous BF16 rank gather.

Reproduce the Blackhole accurate H-reduction tree, including its two positive
zero additions per rank. No collective, caller scale or BF16 output conversion
is changed. Eligible local2560 Q1..32 reductions select this module.
"""

from pathlib import Path

import ttnn

_KERNELS = Path(__file__).parent / "kernels"


def exact_rank_sum(gathered):
    if not isinstance(gathered, ttnn.Tensor) or len(gathered.shape) != 4:
        raise ValueError("Expected a four-dimensional device TTNN tensor")
    device = gathered.device()
    q = gathered.shape[-2]
    tile = gathered.get_tile()
    if (
        device.arch() != ttnn.device.Arch.BLACKHOLE
        or tuple(device.shape) != (1, 4)
        or not 1 <= q <= 32
        or tuple(gathered.shape) != (1, 4, q, 2560)
        or tuple(gathered.padded_shape) != (1, 4, 32, 2560)
        or gathered.dtype != ttnn.bfloat16
        or gathered.layout != ttnn.TILE_LAYOUT
        or gathered.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        or tuple(tile.tile_shape) != (32, 32)
        or tile.transpose_within_face
        or tile.transpose_of_faces
        or not ttnn.is_tensor_storage_on_device(gathered)
        or not gathered.is_allocated()
    ):
        raise ValueError("Expected allocated TP4 BF16 standard TILE DRAM [1,4,Q<=32,2560]")
    output = ttnn.empty(
        (1, 1, q, 2560),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = device.compute_with_storage_grid_size()
    used = min(80, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(used)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        start, end = 80 * i // used, 80 * (i + 1) // used
        reader[core.x][core.y] = [gathered.buffer_address(), start, end - start]
        compute[core.x][core.y] = [end - start]
        writer[core.x][core.y] = [output.buffer_address(), start, end - start]
    specifications = [
        ("reader", ttnn.TensorAccessorArgs(gathered).get_compile_time_args(), reader, ttnn.ReaderConfigDescriptor()),
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
            kernel_source=str(_KERNELS / f"exact_rank_sum_{role}.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for role, ct, rt, config in specifications
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        for index, dtype, page, size in ((0, ttnn.bfloat16, 2048, 8192), (1, ttnn.float32, 4096, 8192))
    ]
    ttnn.generic_op([gathered, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return output
