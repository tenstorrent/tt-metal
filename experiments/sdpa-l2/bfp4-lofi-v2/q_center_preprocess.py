# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent device Q-minus-BF16-bias, live FP32 subtraction, RNE7/BF16."""
from pathlib import Path

import torch

import center_preprocess as CENTER

HERE = Path(__file__).resolve().parent
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/q_center_preprocess/"


def oracle(src, bias):
    # Reuse the ordinary finite input/centered-value contract; conservative
    # BFP4 group bounds are unnecessary here, so validate independently.
    assert src.dtype == bias.dtype == torch.bfloat16
    assert src.ndim == 4 and src.shape[0] == 1 and src.shape[-1] == 128
    assert tuple(bias.shape) == (1, src.shape[1], 32, 128)
    assert torch.equal(bias, bias[:, :, :1].expand_as(bias))
    for x in (src, bias):
        assert bool(torch.isfinite(x).all())
        assert bool(((x == 0) | (x.abs().float() >= 2.0**-126)).all())
    centered = src.float() - bias[:, :, :1].float()
    assert bool(torch.isfinite(centered).all())
    assert bool(((centered == 0) | (centered.abs() >= 2.0**-126)).all())
    raw = centered.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    rounded = ((raw + 0xFFFF + ((raw >> 17) & 1)) & 0xFFFE0000).int().view(torch.float32)
    assert bool(torch.isfinite(rounded).all()), "RNE overflow is outside the contract"
    return rounded


def build(device, src, bias, ncores=1):
    """Return BF16 centered-RNE7 output, invoke, actual core count."""
    import ttnn

    assert src.dtype == bias.dtype == ttnn.bfloat16
    assert len(src.shape) == 4 and src.shape[0] == 1 and src.shape[-1] == 128
    assert src.shape[2] > 0 and src.shape[2] % 32 == 0 and ncores > 0
    assert tuple(bias.shape) == (1, src.shape[1], 32, 128)
    output = ttnn.allocate_tensor_on_device(src.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                          device, ttnn.DRAM_MEMORY_CONFIG)
    tiles, tiles_per_head = src.volume() // 1024, src.shape[2] // 32 * 4
    size = device.compute_with_storage_grid_size()
    ncores = min(ncores, size.x * size.y, tiles // 4)
    coords = [ttnn.CoreCoord(i % size.x, i // size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    cbs = [ttnn.CBDescriptor(total_size=capacity * 2048, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16,
                                                       page_size=2048)])
           for index, capacity in ((0, 8), (1, 4), (16, 8))]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, (offset, count) in zip(coords, CENTER.core_segments(tiles, ncores)):
        reader[core.x][core.y] = [src.buffer_address(), bias.buffer_address(), offset, count]
        writer[core.x][core.y] = [output.buffer_address(), offset, count]
        compute[core.x][core.y] = [offset, count]
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
        ttnn.KernelDescriptor(kernel_source=PREFIX + "reader.cpp", core_ranges=grid,
            compile_time_args=[tiles_per_head] + ttnn.TensorAccessorArgs(src).get_compile_time_args()
            + ttnn.TensorAccessorArgs(bias).get_compile_time_args(), runtime_args=reader,
            config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
            compile_time_args=ttnn.TensorAccessorArgs(output).get_compile_time_args(), runtime_args=writer,
            config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "compute.cpp", core_ranges=grid,
            compile_time_args=[tiles_per_head], runtime_args=compute,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False, math_approx_mode=False)),
    ])
    return output, lambda: ttnn.generic_op([src, bias, output], desc), ncores
