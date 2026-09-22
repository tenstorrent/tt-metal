# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fuse packed gate/up slicing, SwiGLU, and the down-projection input reshard.

The input is the existing DRAM-sharded matmul's BF16 output, with concatenated
gate/up halves. Preserve its intermediate BF16 SiLU rounding before multiply.
This is a local phase of the decode experiment, not a complete decoder program.
"""

from pathlib import Path

import ttnn


def swiglu_program(packed, output):
    if tuple(packed.shape) != (1, 1, 1, 7168) or tuple(output.shape) != (1, 1, 1, 3584):
        raise ValueError("The experimental fused SwiGLU supports batch-one Llama 3.1-8B only")
    for tensor in (packed, output):
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError("Fused SwiGLU requires BF16 tiled tensors")
        if tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            raise ValueError("Fused SwiGLU requires width-sharded L1 tensors")
        if tensor.memory_config().buffer_type != ttnn.BufferType.L1:
            raise ValueError("Fused SwiGLU requires width-sharded L1 tensors")
    shard = output.memory_config().shard_spec
    cores = ttnn.corerange_to_cores(shard.grid, row_wise=True)
    if len(cores) != 16 or list(shard.shape) != [32, 224] or shard.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("Output must be the standard sixteen-core [32,224] down-projection input")
    source = str(Path(__file__).with_name("kernels") / "swiglu.cpp")
    compile_args = list(ttnn.TensorAccessorArgs(packed).get_compile_time_args())
    runtime = ttnn.RuntimeArgs()
    for index, core in enumerate(cores):
        runtime[core.x][core.y] = [packed.buffer_address(), index * 7]
    reader = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=shard.grid,
        compile_time_args=compile_args,
        runtime_args=runtime,
        defines=[("READER", "1")],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=shard.grid,
        defines=[("COMPUTE", "1")],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        ),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=4 * 2048,
            core_ranges=shard.grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (0, 1, 2)
    ]
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(16, output))
    return ttnn.ProgramDescriptor(kernels=[reader, compute], cbs=cbs, semaphores=[])


def fused_swiglu(packed, *, output_memory_config):
    output = ttnn.empty(
        (1, 1, 1, 3584),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=packed.device(),
        memory_config=output_memory_config,
    )
    ttnn.generic_op([packed, output], swiglu_program(packed, output))
    return output
