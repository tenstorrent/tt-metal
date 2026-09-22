# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU directly from a packed ``[gate | up]`` BF16 projection.

This experimental op keeps the accepted matmul unchanged and only replaces the
two DRAM slice operations plus the following binary operation.  Each worker
reads the corresponding gate/up tiles from the packed tensor, rounds SiLU to a
BF16 L1 circular buffer (matching binary_ng's operand-activation boundary), and
writes ``silu(gate) * up``.
"""

import math

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/autoports/qwen_qwen3_8_27b/tt/kernels/packed_swiglu.cpp"
_TILE = ttnn.Tile((32, 32))


def packed_swiglu(packed: ttnn.Tensor) -> ttnn.Tensor:
    """Return ``silu(gate) * up`` for a TILE BF16 ``[gate | up]`` tensor."""
    if packed.dtype != ttnn.bfloat16 or packed.layout != ttnn.TILE_LAYOUT:
        raise ValueError("packed_swiglu requires a TILE BF16 tensor")
    if packed.shape[-1] % 64:
        raise ValueError("packed_swiglu requires two tile-aligned equal halves")

    shape = list(packed.shape)
    shape[-1] //= 2
    output = ttnn.empty(
        shape,
        dtype=packed.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=packed.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # The projection is column-parallel across the mesh.  ttnn.empty defaults
    # to a replicated placement even though generic_op writes distinct local
    # buffers, so preserve the input placement explicitly.
    output.update_tensor_topology(packed.tensor_topology())

    grid = packed.device().compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    core_list = ttnn.corerange_to_cores(cores, row_wise=True)
    half_width_tiles = packed.shape[-1] // 64
    total_output_tiles = math.prod(list(output.padded_shape)) // (32 * 32)
    tiles_per_core = (total_output_tiles + len(core_list) - 1) // len(core_list)

    reader_args = []
    writer_args = []
    compute_args = []
    for index, core in enumerate(core_list):
        start = index * tiles_per_core
        count = min(tiles_per_core, max(0, total_output_tiles - start))
        reader_args.append((core, [start, count]))
        writer_args.append((core, [start, count]))
        compute_args.append((core, [count]))

    tile_bytes = _TILE.get_tile_size(packed.dtype)
    tile_desc = ttnn.TileDescriptor(_TILE)

    def cb(index, pages):
        return ttnn.CBDescriptor(
            total_size=pages * tile_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=packed.dtype,
                    page_size=tile_bytes,
                    tile=tile_desc,
                )
            ],
        )

    reader_ctas = ttnn.TensorAccessorArgs(packed).get_compile_time_args()
    writer_ctas = ttnn.TensorAccessorArgs(output).get_compile_time_args()
    descriptor = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        ncrisc_compile_time_args=reader_ctas,
        brisc_compile_time_args=writer_ctas,
        ncrisc_named_compile_time_args=[("half_width_tiles", half_width_tiles)],
        ncrisc_common_runtime_args=[packed.buffer_address()],
        brisc_common_runtime_args=[output.buffer_address()],
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=reader_args,
            brisc_args=writer_args,
            trisc_args=compute_args,
        ),
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )
    program = ttnn.ProgramDescriptor(
        kernels=descriptor.get_kernel_descriptors().kernels,
        cbs=[cb(0, 8), cb(1, 8), cb(2, 8), cb(3, 2)],
    )
    return ttnn.generic_op([packed, output], program)
