# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Update micro op.

Updates 1x576 KV cache in DRAM with new NOPE (512) and ROPE (64) data.

Input expectations:
  NOPE: Single-core L1 tensor (1x512 on one core).
        The op multicasts this to a 2x8 knope grid (16 cores), then each
        knope core reads 1 DRAM page (1x32), patches its 32-element slice,
        and writes back.
  ROPE: Split across 2 cores (1x32 each, WIDTH_SHARDED).
        Each rope core independently reads 1 DRAM page, patches, writes back.

Uses unified_kernels/kv_cache_update.hpp Op and unified_kernels/mcast.hpp Op.
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# CB indices used by the kernel (must match named compile-time args)
KV_CACHE_INPUT_CB = 31
KV_CACHE_OUTPUT_CB = 30
KV_CACHE_INTERMED_CB = 29
KV_CACHE_INTERMED_SYNC_CB = 28
KV_RMSNORM_OUTPUT_CB = 26
KROPE_OUTPUT_CB = 27
KV_CACHE_NUM_TILES = 1
KROPE_WT = 1
OUTPUT_CB = 0

TILE_32x32 = ttnn.Tile((32, 32))
TILE_16x32 = ttnn.Tile((16, 32))
TILE_1x32 = ttnn.Tile((1, 32))

# NOPE rmsnorm output: 1 tile of 16x32 = 512 elements
KV_RMSNORM_NUM_TILES = 1


class KVCacheUpdate:
    """
    KV Cache Update: mcast nope data -> read DRAM -> untilize -> patch -> tilize -> write DRAM.
    """

    @staticmethod
    def op(
        nope_cache_tensor,
        rope_cache_tensor,
        full_kv_cache_tensor,
        position_ids_tensor: ttnn.Tensor,
        output_tensor,
        knope_grid,
    ):
        """
        Run KV cache update as a standalone program.

        Args:
            nope_cache_tensor: L1 tensor on a SINGLE core containing the rmsnorm output
                               (1x512 bfloat16). This is the mcast source.
            rope_cache_tensor: L1 tensor split across 2 rope cores (1x32 each, WIDTH_SHARDED).
            full_kv_cache_tensor: DRAM tensor for full KV cache (ND-sharded, BFP8).
            position_ids_tensor: L1 tensor with write position index (replicated across all cores).
            output_tensor: Not used for readback; validation reads from full_kv_cache_tensor.
            knope_grid: CoreRangeSet for the 16 knope receiver cores (e.g. 2x8 grid).

        Returns:
            Output of generic_op.
        """
        device = nope_cache_tensor.device()
        nope_sender_grid = nope_cache_tensor.memory_config().shard_spec.grid
        rope_core_grid = rope_cache_tensor.memory_config().shard_spec.grid

        # Full grid: nope sender + knope receivers + rope cores
        full_grid = nope_sender_grid.merge(knope_grid).merge(rope_core_grid)
        # KV cache update grid: knope + rope (cores that do DRAM read/write)
        kv_cache_core_grid = knope_grid.merge(rope_core_grid)

        tensor_accessor_args = ttnn.TensorAccessorArgs(full_kv_cache_tensor)
        ncrisc_compile_time_args = tensor_accessor_args.get_compile_time_args()
        brisc_compile_time_args = tensor_accessor_args.get_compile_time_args()

        # Compute mcast NOC coordinates
        knope_grid_range = list(knope_grid.ranges())[0]
        nope_mcast_dest_start_core = device.worker_core_from_logical_core(knope_grid_range.start)
        nope_mcast_dest_end_core = device.worker_core_from_logical_core(knope_grid_range.end)
        nope_mcast_num_cores = knope_grid_range.grid_size().x * knope_grid_range.grid_size().y
        nope_mcast_num_dests = (
            nope_mcast_num_cores - 1 if knope_grid_range.contains(nope_sender_grid) else nope_mcast_num_cores
        )

        kv_rmsnorm_page_size = TILE_16x32.get_tile_size(ttnn.bfloat16)
        nope_mcast_data_size_bytes = KV_RMSNORM_NUM_TILES * kv_rmsnorm_page_size

        ncrisc_named = [
            ("kv_rmsnorm_output_cb", KV_RMSNORM_OUTPUT_CB),
            ("krope_output_cb", KROPE_OUTPUT_CB),
            ("kv_cache_intermed_cb", KV_CACHE_INTERMED_CB),
            ("kv_cache_intermed_sync_cb", KV_CACHE_INTERMED_SYNC_CB),
            ("kv_cache_output_cb", KV_CACHE_OUTPUT_CB),
            ("kv_cache_grid_start_y", list(rope_core_grid.ranges())[0].start.y),
            ("kv_cache_cur_pos_ready_semaphore_id", 0),
            ("nope_mcast_dest_noc_start_x", nope_mcast_dest_start_core.x),
            ("nope_mcast_dest_noc_start_y", nope_mcast_dest_start_core.y),
            ("nope_mcast_dest_noc_end_x", nope_mcast_dest_end_core.x),
            ("nope_mcast_dest_noc_end_y", nope_mcast_dest_end_core.y),
            ("nope_mcast_sender_semaphore_id", 1),
            ("nope_mcast_receiver_semaphore_id", 2),
            ("nope_mcast_data_size_bytes", nope_mcast_data_size_bytes),
            ("nope_mcast_num_dests", nope_mcast_num_dests),
            ("kv_rmsnorm_num_tiles", KV_RMSNORM_NUM_TILES),
        ]
        brisc_named = [
            ("kv_cache_input_cb", KV_CACHE_INPUT_CB),
            ("kv_cache_grid_start_y", list(rope_core_grid.ranges())[0].start.y),
            ("kv_rmsnorm_output_cb", KV_RMSNORM_OUTPUT_CB),
        ]
        trisc_named = [
            ("kv_cache_input_cb", KV_CACHE_INPUT_CB),
            ("kv_cache_output_cb", KV_CACHE_OUTPUT_CB),
            ("kv_cache_intermed_cb", KV_CACHE_INTERMED_CB),
            ("kv_cache_intermed_sync_cb", KV_CACHE_INTERMED_SYNC_CB),
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
        kv_cache_intermed_sync_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=KV_CACHE_INTERMED_SYNC_CB,
            data_format=ttnn.bfloat16,
            page_size=intermed_page_size,
            tile=ttnn.TileDescriptor(TILE_32x32),
        )

        # Expand kv_rmsnorm_output_cb to cover nope sender + knope grid
        kv_rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            KV_RMSNORM_OUTPUT_CB,
            nope_cache_tensor,
            core_ranges=nope_sender_grid.merge(knope_grid),
        )

        cbs = [
            kv_rmsnorm_output_cb_descriptor,
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
                total_size=KV_CACHE_NUM_TILES * intermed_page_size,
                core_ranges=kv_cache_core_grid,
                format_descriptors=[kv_cache_intermed_cb_format, kv_cache_intermed_sync_cb_format],
            ),
        ]

        mla_kv_cache_cur_pos_ready_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=0,
            core_ranges=kv_cache_core_grid,
            initial_value=0,
        )

        # Mcast semaphores: sender (BRISC) and receiver (NCRISC)
        nope_mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=1,
            core_ranges=full_grid,
            initial_value=0,
        )
        nope_mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=2,
            core_ranges=full_grid,
            initial_value=0,
        )

        pos_addr = position_ids_tensor.buffer_address()
        ncrisc_common_runtime_args = [full_kv_cache_tensor.buffer_address(), pos_addr]
        brisc_common_runtime_args = [full_kv_cache_tensor.buffer_address(), pos_addr]

        knope_grid_width = knope_grid_range.grid_size().x
        knope_core_index_desc = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="knope_core_index",
            core_values=[
                (
                    ttnn.CoreCoord(x, y),
                    (y - knope_grid_range.start.y) * knope_grid_width + (x - knope_grid_range.start.x),
                )
                for y in range(knope_grid_range.start.y, knope_grid_range.end.y + 1)
                for x in range(knope_grid_range.start.x, knope_grid_range.end.x + 1)
            ],
            other_value=0,
        )
        kernel_desc = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/kv_cache_update/kernels/kv_cache_update_kernel.cpp",
            core_ranges=full_grid,
            ncrisc_compile_time_args=ncrisc_compile_time_args,
            brisc_compile_time_args=brisc_compile_time_args,
            ncrisc_named_compile_time_args=ncrisc_named,
            brisc_named_compile_time_args=brisc_named,
            trisc_named_compile_time_args=trisc_named,
            ncrisc_common_runtime_args=ncrisc_common_runtime_args,
            brisc_common_runtime_args=brisc_common_runtime_args,
            per_core_compile_time_descriptors=[knope_core_index_desc],
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_nope_sender_core",
                    core_range=nope_sender_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_nope_core",
                    core_range=knope_grid,
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
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        )
        kernel_result = kernel_desc.get_kernel_descriptors()
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_result.kernels,
            cbs=cbs,
            semaphores=[
                mla_kv_cache_cur_pos_ready_semaphore_descriptor,
                nope_mcast_sender_semaphore_descriptor,
                nope_mcast_receiver_semaphore_descriptor,
            ],
        )
        io_tensors = [nope_cache_tensor, rope_cache_tensor, position_ids_tensor, output_tensor]

        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
