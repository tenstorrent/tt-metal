# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Raw-tile GU permutation for two contiguous readers per DRAM bank."""
from pathlib import Path
import ttnn


def split_gu_bank_rows(weight, mesh):
    if tuple(weight.shape) != (4096, 7168) or weight.dtype != ttnn.bfloat4_b:
        raise ValueError("Expected the fixed Llama packed GU BFP4 weight")
    old = weight.memory_config()
    if old.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED or old.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("Expected eight width-sharded DRAM banks")
    if list(old.shard_spec.shape) != [4096, 896] or weight.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Expected tile layout and eight 4096x896 bank shards")
    memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
        ttnn.ShardSpec(old.shard_spec.grid, [8192, 448], old.shard_spec.orientation))
    output = ttnn.empty((8192, 3584), dtype=weight.dtype, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=memory)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    rt = ttnn.RuntimeArgs()
    for bank in range(8):
        rt[bank][0] = [bank, weight.buffer_address(), output.buffer_address()]
    kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name("kernels") / "repack_gu.cpp"),
        core_ranges=cores, runtime_args=rt,
        compile_time_args=[*ttnn.TensorAccessorArgs(weight).get_compile_time_args(),
                           *ttnn.TensorAccessorArgs(output).get_compile_time_args()],
        config=ttnn.ReaderConfigDescriptor())
    program = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[ttnn.CBDescriptor(
        total_size=16 * 28 * 576, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat4_b, page_size=576)])])
    ttnn.generic_op([weight, output], program)
    return output
