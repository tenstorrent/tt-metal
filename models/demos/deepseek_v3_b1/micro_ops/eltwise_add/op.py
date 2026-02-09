# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Eltwise Add Operation with Per-Core Indexing.

Used after down_proj in MoE kernel to add fused_add tensor.

Tensor layout:
- in0 (down_proj_out): 1x896 per core, WIDTH_SHARDED, 1x32 tiles
- in1 (fused_add): 1x7168 per core (replicated), HEIGHT_SHARDED, 1x32 tiles
- out: 1x896 per core, WIDTH_SHARDED, 1x32 tiles

CB view: 32x32 tiles (aliasing from 1x32 tensor tiles)
- 896 elements = 28 tiles of 1x32, viewed as ~1 tile of 32x32
- Last 128 elements are garbage padding (ignored in validation)

Each core uses sender_index to offset into fused_add:
- Core 0 (sender_index=0): fused_add[0:896]
- Core 1 (sender_index=1): fused_add[896:1792]
- etc.

Core logic is in unified_kernels/eltwise_add.hpp.
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor


class EltwiseAdd:
    """
    Eltwise add with per-core indexing using ttnn.generic_op.
    """

    @staticmethod
    def golden(
        down_proj_out: torch.Tensor,
        fused_add: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch reference implementation.

        Args:
            down_proj_out: Output from down_proj [1, 1, 1, total_width]
            fused_add: Tensor to add [1, 1, 1, total_width]

        Returns:
            down_proj_out + fused_add
        """
        return down_proj_out + fused_add

    @staticmethod
    def op(
        down_proj_out_tensor: ttnn.Tensor,
        fused_add_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Execute eltwise add with per-core indexing.

        Args:
            down_proj_out_tensor: WIDTH_SHARDED, 1x896 per core, 1x32 tiles
            fused_add_tensor: HEIGHT_SHARDED (replicated), 1x7168 per core, 1x32 tiles
            output_tensor: WIDTH_SHARDED, 1x896 per core, 1x32 tiles

        Returns:
            Output tensor with down_proj_out + fused_add (indexed per core)
        """
        # Get tensor info
        down_proj_dtype = down_proj_out_tensor.dtype

        # Get shard shapes
        down_proj_shard_shape = down_proj_out_tensor.memory_config().shard_spec.shape
        fused_add_shard_shape = fused_add_tensor.memory_config().shard_spec.shape

        # Dimensions
        width_per_core = down_proj_shard_shape[1]  # 896
        total_width = fused_add_shard_shape[1]  # 7168

        # Element size
        if down_proj_dtype == ttnn.bfloat16:
            element_size_bytes = 2
        else:
            raise ValueError(f"Unsupported dtype: {down_proj_dtype}")

        slice_size_bytes = width_per_core * element_size_bytes  # 896 * 2 = 1792

        logger.debug(f"width_per_core={width_per_core}, total_width={total_width}, slice_size_bytes={slice_size_bytes}")

        # Get compute cores from down_proj_out tensor (WIDTH_SHARDED)
        compute_core_grid = down_proj_out_tensor.memory_config().shard_spec.grid
        compute_cores_list = ttnn.corerange_to_cores(compute_core_grid, row_wise=True)
        num_cores = len(compute_cores_list)
        # print(compute_cores_list)

        logger.debug(f"num_cores={num_cores}")

        # 32x32 tile for CB view (just changes interpretation, not actual size)
        tile_32x32 = ttnn.Tile([32, 32])
        tile_32x32_desc = ttnn.TileDescriptor(tile_32x32)

        # CB indices
        cb_in0 = 0  # down_proj output (32x32 view, backed by tensor)
        cb_in1 = 1  # fused_add (32x32 view, backed by tensor, read ptr updated to offset)
        cb_out = 2  # output (32x32 view, backed by tensor)

        # Number of 32x32 tiles (for CB view)
        num_tiles_32x32 = 1  # We view the data as 1 tile of 32x32

        # Actual data sizes
        # down_proj: 896 * 2 = 1792 bytes (input, backed by tensor)
        # fused_add: 7168 * 2 = 14336 bytes per core (replicated, backed by tensor)
        # output: 32x32 tile = 2048 bytes (pack_tile writes full 32x32 tile)
        down_proj_size_bytes = slice_size_bytes  # 1792
        fused_add_size_bytes = total_width * element_size_bytes  # 14336
        tile_32x32_size_bytes = 32 * 32 * element_size_bytes  # 2048

        # ========== CIRCULAR BUFFERS ==========
        # total_size = actual data size, tile = 32x32 view (just changes interpretation)

        # CB 0: down_proj_out - 32x32 view, backed by tensor
        cb0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, down_proj_out_tensor)
        cb0_descriptor.total_size = down_proj_size_bytes
        cb0_descriptor.format_descriptors[0].tile = tile_32x32_desc
        cb0_descriptor.format_descriptors[0].page_size = down_proj_size_bytes

        # CB 1: fused_add - 32x32 view, backed by tensor
        # TRISC will update read pointer to indexed offset (no copy needed!)
        cb1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in1, fused_add_tensor)
        cb1_descriptor.total_size = fused_add_size_bytes  # full fused_add tensor size
        cb1_descriptor.format_descriptors[0].tile = tile_32x32_desc
        cb1_descriptor.format_descriptors[0].page_size = down_proj_size_bytes  # page_size = slice size for reading

        # CB 2: output - 32x32 tile, backed by tensor
        # pack_tile writes full 32x32 tile, so CB needs 2048 bytes
        cb2_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)
        cb2_descriptor.total_size = tile_32x32_size_bytes
        cb2_descriptor.format_descriptors[0].tile = tile_32x32_desc
        cb2_descriptor.format_descriptors[0].page_size = tile_32x32_size_bytes

        # ========== Per-core sender_index values ==========
        sender_index_core_values = []
        for idx, core in enumerate(compute_cores_list):
            sender_index_core_values.append((core, idx))

        # ========== KERNEL ==========
        # Number of pages for CB sharded buffer setup (total_size / page_size)
        cb_in0_wait_tiles = down_proj_size_bytes // down_proj_size_bytes  # 1792 / 1792 = 1
        cb_in1_wait_tiles = fused_add_size_bytes // down_proj_size_bytes  # 14336 / 1792 = 8

        # Common compile-time args
        common_compile_time_args = [
            # CB indices
            ("add_cb_in0", cb_in0),
            ("add_cb_in1", cb_in1),
            ("add_cb_out", cb_out),
            # Dimensions
            ("add_num_tiles", num_tiles_32x32),
            ("add_slice_size_bytes", slice_size_bytes),
            # Number of pages to push/wait for sharded buffers
            ("add_cb_in0_wait_tiles", cb_in0_wait_tiles),  # 1 page
            ("add_cb_in1_wait_tiles", cb_in1_wait_tiles),  # 8 pages
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/eltwise_add/eltwise_add_kernel.cpp",
            core_ranges=compute_core_grid,
            ncrisc_named_compile_time_args=common_compile_time_args,
            brisc_named_compile_time_args=common_compile_time_args,
            trisc_named_compile_time_args=common_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="add_sender_index",
                    core_values=sender_index_core_values,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[cb0_descriptor, cb1_descriptor, cb2_descriptor],
            semaphores=[],
        )

        # Execute generic op
        io_tensors = [down_proj_out_tensor, fused_add_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
