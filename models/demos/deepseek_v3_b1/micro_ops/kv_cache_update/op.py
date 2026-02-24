# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Update micro op.

Reads 16 BFP8 tiles from DRAM into L1, untilize (BFP8->bfloat16), tilize (bfloat16->BFP8).
Runs on a grid with nope cores (kv_rmsnorm) and rope cores (krope). Uses kv_cache_update.hpp Op.
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# CB indices used by the kernel (must match named compile-time args)
KV_CACHE_INPUT_CB = 31
KV_CACHE_OUTPUT_CB = 30
KV_CACHE_INTERMED_CB = 29
KV_RMSNORM_OUTPUT_CB = 26
KROPE_OUTPUT_CB = 27
KV_CACHE_NUM_TILES = 16
KROPE_WT = 1
OUTPUT_CB = 0

TILE_32x32 = ttnn.Tile((32, 32))
TILE_16x32 = ttnn.Tile((16, 32))
TILE_1x32 = ttnn.Tile((1, 32))


class KVCacheUpdate:
    """
    KV Cache Update: read DRAM -> untilize -> tilize. Uses unified_kernels/kv_cache_update.hpp.
    """

    @staticmethod
    def op(
        nope_cache_tensor,
        rope_cache_tensor,
        full_kv_cache_tensor,
        position_ids_tensor: ttnn.Tensor,
        output_tensor,  # not used
    ):
        """
        Run KV cache update as a standalone program.

        Args:
            nope_cache_tensor: L1 tensor backing NOPE_CACHE_CB (tilized BFP8 input).
            rope_cache_tensor: DRAM tensor for rope cache (buffer address + tensor accessor).
            full_kv_cache_tensor: DRAM tensor for full KV cache (buffer address + tensor accessor).
            position_id: Write position index.
            output_tensor: not used, validation is read back from full_kv_cache_tensor

        Returns:
            Output of generic_op (includes output_tensor when provided).
        """
        nope_core_grid = nope_cache_tensor.memory_config().shard_spec.grid
        rope_core_grid = rope_cache_tensor.memory_config().shard_spec.grid
        kv_cache_core_grid = nope_core_grid.merge(rope_core_grid)

        tensor_accessor_args = ttnn.TensorAccessorArgs(full_kv_cache_tensor)
        ncrisc_compile_time_args = tensor_accessor_args.get_compile_time_args()
        brisc_compile_time_args = tensor_accessor_args.get_compile_time_args()

        device = full_kv_cache_tensor.device()
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1)
        )
        full_grid_mcast_start_core = device.worker_core_from_logical_core(full_device_grid.start)
        full_grid_mcast_end_core = device.worker_core_from_logical_core(full_device_grid.end)
        full_grid_mcast_num_dests = full_device_grid.grid_size().x * full_device_grid.grid_size().y

        # CB indices and krope_Wt passed as named compile-time args; kernel uses get_named_compile_time_arg_val
        ncrisc_named = [
            ("kv_rmsnorm_output_cb", KV_RMSNORM_OUTPUT_CB),
            ("krope_output_cb", KROPE_OUTPUT_CB),
        ]
        brisc_named = [
            ("kv_cache_input_cb", KV_CACHE_INPUT_CB),
            ("kv_cache_output_cb", KV_CACHE_OUTPUT_CB),
            ("kv_cache_intermed_cb", KV_CACHE_INTERMED_CB),
            ("kv_rmsnorm_output_cb", KV_RMSNORM_OUTPUT_CB),
            ("krope_output_cb", KROPE_OUTPUT_CB),
            ("kv_cache_grid_start_y", list(rope_core_grid.ranges())[0].start.y),
            ("full_grid_mcast_start_x", full_grid_mcast_start_core.x),
            ("full_grid_mcast_start_y", full_grid_mcast_start_core.y),
            ("full_grid_mcast_end_x", full_grid_mcast_end_core.x),
            ("full_grid_mcast_end_y", full_grid_mcast_end_core.y),
            ("full_grid_mcast_num_dests", full_grid_mcast_num_dests - 1),
            ("kv_cache_cur_pos_ready_semaphore_id", 0),
        ]
        trisc_named = [
            ("kv_cache_input_cb", KV_CACHE_INPUT_CB),
            ("kv_cache_output_cb", KV_CACHE_OUTPUT_CB),
            ("kv_cache_intermed_cb", KV_CACHE_INTERMED_CB),
            ("kv_rmsnorm_output_cb", KV_RMSNORM_OUTPUT_CB),
            ("krope_output_cb", KROPE_OUTPUT_CB),
        ]

        kv_cache_page_size = TILE_32x32.get_tile_size(ttnn.bfloat8_b)
        intermed_page_size = TILE_32x32.get_tile_size(ttnn.bfloat16)

        kv_cache_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=KV_CACHE_INPUT_CB,
            data_format=ttnn.bfloat8_b,
            page_size=kv_cache_page_size,
            tile=ttnn.TileDescriptor(TILE_32x32),
        )
        kv_cache_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=KV_CACHE_OUTPUT_CB,
            data_format=ttnn.bfloat8_b,
            page_size=kv_cache_page_size,
            tile=ttnn.TileDescriptor(TILE_32x32),
        )
        kv_cache_intermed_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=KV_CACHE_INTERMED_CB,
            data_format=ttnn.bfloat16,
            page_size=intermed_page_size,
            tile=ttnn.TileDescriptor(TILE_32x32),
        )

        kv_rmsnorm_output_cb_format = ttnn.cb_descriptor_from_sharded_tensor(KV_RMSNORM_OUTPUT_CB, nope_cache_tensor)

        # Output CB: tensor-backed on nope core when output_tensor provided, else L1-only
        cbs = [
            kv_rmsnorm_output_cb_format,
            ttnn.cb_descriptor_from_sharded_tensor(KROPE_OUTPUT_CB, rope_cache_tensor),
            ttnn.cb_descriptor_from_sharded_tensor(OUTPUT_CB, output_tensor),
            ttnn.CBDescriptor(
                total_size=KV_CACHE_NUM_TILES * kv_cache_page_size,
                core_ranges=kv_cache_core_grid,
                format_descriptors=[kv_cache_input_cb_format],
            ),
            ttnn.CBDescriptor(
                total_size=KV_CACHE_NUM_TILES * kv_cache_page_size,
                core_ranges=kv_cache_core_grid,
                format_descriptors=[kv_cache_output_cb_format],
            ),
            ttnn.CBDescriptor(
                total_size=(KV_CACHE_NUM_TILES + 1) * intermed_page_size,
                core_ranges=kv_cache_core_grid,
                format_descriptors=[kv_cache_intermed_cb_format],
            ),
        ]

        # not used in unit test, but needed for fused sdpa to wait on kv cache update
        mla_kv_cache_cur_pos_ready_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=0,
            core_ranges=kv_cache_core_grid,
            initial_value=0,
        )
        pos_addr = position_ids_tensor.buffer_address()
        ncrisc_common_runtime_args = [full_kv_cache_tensor.buffer_address()]
        brisc_common_runtime_args = [full_kv_cache_tensor.buffer_address(), pos_addr]

        kernel_desc = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/kv_cache_update/kernels/kv_cache_update_kernel.cpp",
            core_ranges=kv_cache_core_grid,
            ncrisc_compile_time_args=ncrisc_compile_time_args,
            brisc_compile_time_args=brisc_compile_time_args,
            ncrisc_named_compile_time_args=ncrisc_named,
            brisc_named_compile_time_args=brisc_named,
            trisc_named_compile_time_args=trisc_named,
            ncrisc_common_runtime_args=ncrisc_common_runtime_args,
            brisc_common_runtime_args=brisc_common_runtime_args,
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_nope_core",
                    core_range=nope_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_rope_core",
                    core_range=rope_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
            #  noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        )
        kernel_result = kernel_desc.get_kernel_descriptors()
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_result.kernels,
            cbs=cbs,
            semaphores=[mla_kv_cache_cur_pos_ready_semaphore_descriptor],
        )
        io_tensors = [nope_cache_tensor, rope_cache_tensor, position_ids_tensor, output_tensor]

        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
