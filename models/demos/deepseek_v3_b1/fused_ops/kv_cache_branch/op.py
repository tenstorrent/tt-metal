# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.rope.op import RopeSingleCore
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class KVCacheBranch:
    """
    KV Cache Branch fused operation implementation using ttnn.generic_op.

    This class implements the KV cache branch operations as a fused execution:
    """

    @staticmethod
    def golden(
        input_tensor,
        W_dkv_rope_tensor,
        gamma_tensor,
        cos_tensor,
        sin_tensor,
        position_ids_tensor,
        epsilon=1e-6,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            W_dkv_rope_tensor: W_dkv_rope tensor (torch.Tensor)

        Returns:
            Output tensor with KV cache branch operations applied
        """

        def rmsnorm(x, gamma):
            variance = x.pow(2).mean(-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + epsilon)
            return normalized * gamma

        nope = 512
        rope = 64
        compressed_kv = input_tensor @ W_dkv_rope_tensor
        compressed_kv, k_rope = torch.split(compressed_kv, [nope, rope], dim=-1)
        kv = rmsnorm(compressed_kv, gamma_tensor)
        k_rope = RopeSingleCore.golden(k_rope, cos_tensor, sin_tensor, position_ids_tensor).squeeze((0, 1))

        full_kv_cache_tensor = torch.cat([kv, k_rope], dim=-1)
        return full_kv_cache_tensor

    @staticmethod
    def op(
        input_tensor,
        dkv_matmul_weights_tensor,
        gamma_tensor,
        cos_tensor,
        sin_tensor,
        trans_mat_tensor,
        output_tensor,
        kv_cache_tensor,
        position_ids_tensor,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
    ):
        """
        Execute KV cache branch fused operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded)
            dkv_matmul_weights_tensor: DKV Matmul weights tensor (must be width sharded)
            gamma_tensor: Gamma tensor (must be sharded)
            cos_tensor: Cos tensor (must be sharded)
            sin_tensor: Sin tensor (must be sharded)
            kv_cache_tensor: Optional KV cache tensor in DRAM (interleaved) to write results to
            position_ids_tensor: Sequence position index to write to in KV cache
            epsilon: Epsilon for RMSNorm
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with KV cache branch operations applied
        """
        # Get tensor properties
        data_format = input_tensor.dtype

        # Get core grid from input tensor's memory config
        input_core_grid = input_tensor.memory_config().shard_spec.grid

        device = input_tensor.device()

        # DKV Matmul (9x2)
        dkv_matmul_weights_memory_config = dkv_matmul_weights_tensor.memory_config()
        dkv_matmul_weights_core_grid = dkv_matmul_weights_memory_config.shard_spec.grid

        # Calculate per-core width in tiles for matmul (from shard spec)
        # Get shard width directly from shard_spec and divide by tile width from tensor
        dkv_matmul_weights_tile = dkv_matmul_weights_tensor.get_tile()
        dkv_matmul_weights_shard_shape = dkv_matmul_weights_memory_config.shard_spec.shape
        dkv_matmul_weights_shard_width = dkv_matmul_weights_shard_shape[1]  # Width dimension
        dkv_matmul_out_w = (
            dkv_matmul_weights_shard_width // dkv_matmul_weights_tile.tile_shape[1]
        )  # Per-core width in tiles

        # Semaphore IDs for gather synchronization
        # Senders on NCRISC use NOC_0, receiver on BRISC uses NOC_1
        # Only use noc0 semaphore since senders are on NOC_0 (default for NCRISC)
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        kv_numel = 512
        kv_rmsnorm_num_tiles = kv_numel // (16 * 32)  # 512 / 512 = 1 tile (16x32)
        inv_sqrt_numel = 1.0 / math.sqrt(float(kv_numel))
        kv_scalar_packed = float_to_uint32(inv_sqrt_numel)
        epsilon_packed = float_to_uint32(epsilon)

        # KV Cache tensor setup
        # Tile size is now derived from output CB in kernel using get_tile_size()
        kv_cache_buffer_addr = kv_cache_tensor.buffer_address()
        position_ids_tensor_addr = position_ids_tensor.buffer_address()
        kv_cache_tile = kv_cache_tensor.get_tile()
        # Calculate starting tile ID based on write index
        # KV cache shape is [1, 1, seq_len, kv_dim], tiles are [32, 32]

        # CB indices
        # CONSOLIDATE!!!!!!!!!
        # Tile sizes: 1x32 = 64 bytes (BF16), 16x32 = 1024 bytes (BF16), 32x32 = 2048 bytes (BF16)

        cos_cb = 0  # tile (NCRISC reads from DRAM)
        sin_cb = 1  # tile (NCRISC reads from DRAM)
        trans_mat_cb = 2  # 1x32 tile, 64 bytes (sharded, 1 tile per core) - actually 32x32 for matmul
        rotated_input_interm_cb = 3  # 1x32 tile, 64 bytes (Wt tiles, intermediate)
        cos_interm_cb = 4  # 1x32 tile, 64 bytes (Wt tiles, intermediate)
        sin_interm_cb = 5  # 1x32 tile, 64 bytes (Wt tiles, intermediate)
        dkv_matmul_input_cb = 6  # 1x32 tile, 64 bytes (224 tiles = 1x7168)
        dkv_matmul_output_cb = 7  # 1x32 tile, 64 bytes (1 tile per core for rope input)
        dkv_matmul_weights_cb = 8  # 32x32 tile, 2048 bytes (sharded weights)
        kv_rmsnorm_input_cb = 9  # 16x32 tile, 1024 bytes (gathered data, 1 tile)
        kv_rmsnorm_gamma_cb = 10  # 16x32 tile, 1024 bytes (sharded gamma, 1 tile)
        kv_rmsnorm_output_cb = 11  # 16x32 tile, 1024 bytes (sharded output, 1 tile)
        k_rope_output_cb = 12  # 1x32 tile, 64 bytes (Wt tiles output) - SAME AS KV in merged

        # DKV Matmul
        dkv_matmul_k_num_tiles = 7168 // 32
        # Note: kv_rmsnorm_num_tiles already defined above (line 118) using 16x32 tiles
        # kv_rmsnorm_num_faces = 2 for 16x32 tiles (1 * 2 = 2 faces)
        TILE_1x32 = ttnn.Tile((1, 32))
        dkv_matmul_input_page_size = TILE_1x32.get_tile_size(input_tensor.dtype)
        dkv_matmul_ncrisc_named_compile_time_args = [
            ("dkv_matmul_in0", dkv_matmul_input_cb),
            ("dkv_matmul_in1", dkv_matmul_weights_cb),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]

        dkv_matmul_trisc_named_compile_time_args = [
            ("dkv_matmul_in0", dkv_matmul_input_cb),
            ("dkv_matmul_in1", dkv_matmul_weights_cb),
            ("dkv_matmul_out", dkv_matmul_output_cb),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]

        # KV RMSNorm
        # RMSNorm compute compile-time args (named args for TRISC)
        rmsnorm_compute_named_compile_time_args = [
            ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
            ("rmsnorm_rsqrt_fast_approx", 0),
        ]

        kv_rmsnorm_brisc_named_compile_time_args = [
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", kv_rmsnorm_num_tiles),
        ]

        kv_rmsnorm_ncrisc_named_compile_time_args = [
            ("kv_rmsnorm_input_cb", kv_rmsnorm_input_cb),
            ("kv_rmsnorm_gamma_cb", kv_rmsnorm_gamma_cb),
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", kv_rmsnorm_num_tiles),
        ]
        kv_rmsnorm_trisc_named_compile_time_args = [
            ("kv_rmsnorm_input_cb", kv_rmsnorm_input_cb),
            ("kv_rmsnorm_gamma_cb", kv_rmsnorm_gamma_cb),
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", kv_rmsnorm_num_tiles),
        ]

        # KV cache tile size is now derived from output CB in kernel using get_tile_size()
        # ========================================================================
        # Gather setup: k nope matmul cores (senders) -> kv rmsnorm core (receiver)
        # Sender runs on NCRISC (NOC_0 default), Receiver runs on BRISC (NOC_1 default)
        # ========================================================================
        dkv_gather_receiver_core = gamma_tensor.memory_config().shard_spec.grid.ranges()[0].start
        krope_core_grid = trans_mat_tensor.memory_config().shard_spec.grid
        dkv_gather_sender_grid = dkv_matmul_weights_core_grid.subtract(krope_core_grid)

        # Get NOC coordinates for gather destination (receiver core)
        dkv_gather_dest_noc_core = device.worker_core_from_logical_core(dkv_gather_receiver_core)

        # Get number of sender cores (matmul grid)
        dkv_gather_sender_cores_list = ttnn.corerange_to_cores(dkv_gather_sender_grid, row_wise=True)
        dkv_gather_num_senders = len(dkv_gather_sender_cores_list)

        # All senders use NOC_0 (default for NCRISC), so noc0_num_senders = all, noc1_num_senders = 0
        dkv_gather_noc0_num_senders = dkv_gather_num_senders
        dkv_gather_noc1_num_senders = 0

        # Get sender grid dimensions for computing per-core offset in kernel
        # Use logical coordinates since kernel uses UnifiedCoreDescriptor with my_logical_x_/y_
        dkv_gather_sender_grid_ranges = list(dkv_gather_sender_grid.ranges())
        dkv_gather_sender_grid_range = dkv_gather_sender_grid_ranges[0]
        dkv_gather_sender_grid_start_x = dkv_gather_sender_grid_range.start.x
        dkv_gather_sender_grid_start_y = dkv_gather_sender_grid_range.start.y
        dkv_gather_sender_grid_end_x = dkv_gather_sender_grid_range.end.x
        dkv_gather_sender_grid_end_y = dkv_gather_sender_grid_range.end.y

        # Gather sender compile-time args (named args for NCRISC on matmul cores)
        # SenderCTArgs: dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id
        # Plus grid info for computing per-core offset
        dkv_gather_src_num_pages = dkv_matmul_out_w  # dkv matmul output tiles per core (must match matmul cb_push_back)
        dkv_gather_data_size_bytes = dkv_gather_src_num_pages * dkv_matmul_input_page_size
        dkv_gather_sender_named_compile_time_args = [
            ("dkv_gather_dest_noc_x", dkv_gather_dest_noc_core.x),
            ("dkv_gather_dest_noc_y", dkv_gather_dest_noc_core.y),
            ("dkv_gather_data_size_bytes", dkv_gather_data_size_bytes),
            ("dkv_gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("dkv_gather_src_cb", dkv_matmul_output_cb),  # Source CB for gather (dkv matmul output)
            ("dkv_gather_src_num_pages", dkv_gather_src_num_pages),
            ("dkv_gather_sender_grid_start_x", dkv_gather_sender_grid_start_x),
            ("dkv_gather_sender_grid_start_y", dkv_gather_sender_grid_start_y),
            ("dkv_gather_sender_grid_end_x", dkv_gather_sender_grid_end_x),
            ("dkv_gather_sender_grid_end_y", dkv_gather_sender_grid_end_y),
            ("dkv_gather_row_major", 1),  # 1 = row-major linearization
            ("dkv_gather_dst_cb", kv_rmsnorm_input_cb),  # Destination CB: write directly to kv_rmsnorm_input_cb
        ]

        # Gather receiver compile-time args (named args for BRISC on kv rmsnorm core)
        # ReceiverCTArgs: noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id
        # Plus destination CB info for reserve/push
        # Writes directly to kv_rmsnorm_input_cb
        dkv_gather_receiver_named_compile_time_args = [
            ("dkv_gather_noc0_num_senders", dkv_gather_noc0_num_senders),
            ("dkv_gather_noc1_num_senders", dkv_gather_noc1_num_senders),
            ("dkv_gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("dkv_gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("dkv_gather_dst_cb", kv_rmsnorm_input_cb),
            ("dkv_gather_dst_num_pages", dkv_gather_src_num_pages),
        ]

        # ROPE
        rope_tile = cos_tensor.get_tile()
        rope_tile_size = rope_tile.get_tile_size(data_format)
        cos_tensor_address = cos_tensor.buffer_address()
        sin_tensor_address = sin_tensor.buffer_address()

        krope_Wt = 1
        krope_Ht = 1
        num_rope_cores = krope_core_grid.num_cores()
        total_Wt = krope_Wt * num_rope_cores
        rope_cores = ttnn.corerange_to_cores(krope_core_grid)
        start_tile_offset_core_values = [(core, idx * krope_Wt) for idx, core in enumerate(rope_cores)]

        krope_brisc_named_compile_time_args = [
            ("k_rope_output_cb", k_rope_output_cb),
            ("Wt", krope_Wt),
            ("Ht", krope_Ht),
        ]
        krope_ncrisc_named_compile_time_args = [
            ("in_cb", dkv_matmul_output_cb),
            ("cos_cb", cos_cb),
            ("sin_cb", sin_cb),
            ("cos_tensor_address", cos_tensor_address),
            ("sin_tensor_address", sin_tensor_address),
            ("position_ids_tensor_address", position_ids_tensor_addr),
            ("trans_mat_cb", trans_mat_cb),
            ("Wt", krope_Wt),
            ("Ht", krope_Ht),
            ("cos_sin_page_size", rope_tile_size),
            ("total_Wt", total_Wt),
        ]
        krope_trisc_named_compile_time_args = [
            ("in_cb", dkv_matmul_output_cb),
            ("cos_cb", cos_cb),
            ("sin_cb", sin_cb),
            ("trans_mat_cb", trans_mat_cb),
            ("rotated_in_interm_cb", rotated_input_interm_cb),
            ("cos_interm_cb", cos_interm_cb),
            ("sin_interm_cb", sin_interm_cb),
            ("out_cb", k_rope_output_cb),
            ("Wt", krope_Wt),
            ("Ht", krope_Ht),
        ]

        # Create tile descriptor for proper tile dimensions

        # CB X: DKV Matmul input buffer (1x7168 with 1x32 tiles = 224 tiles)
        dkv_matmul_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dkv_matmul_input_cb, input_tensor)
        # CB X: DKV Matmul output buffers
        dkv_matmul_output_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
        dkv_matmul_output_page_size = TILE_1x32.get_tile_size(data_format)
        dkv_matmul_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dkv_matmul_output_cb,
            data_format=data_format,
            page_size=dkv_matmul_output_page_size,
            tile=dkv_matmul_output_tile_descriptor,
        )
        dkv_matmul_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=dkv_matmul_output_page_size,
            core_ranges=dkv_matmul_weights_core_grid,
            format_descriptors=[dkv_matmul_output_cb_format],
        )

        # CB X: DKV Matmul weights buffer
        dkv_matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            dkv_matmul_weights_cb, dkv_matmul_weights_tensor
        )

        # CB X: KV RMSNorm input buffer (on rmsnorm core, receives gathered data)
        TILE_16x32 = ttnn.Tile((16, 32))
        kv_rmsnorm_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
        kv_rmsnorm_page_size = TILE_16x32.get_tile_size(input_tensor.dtype)
        kv_rmsnorm_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=kv_rmsnorm_input_cb,
            data_format=data_format,
            page_size=kv_rmsnorm_page_size,
            tile=kv_rmsnorm_tile_descriptor,
        )
        kv_rmsnorm_input_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * kv_rmsnorm_page_size,
            core_ranges=dkv_gather_sender_grid,
            format_descriptors=[kv_rmsnorm_input_cb_format],
        )

        # CB X: KV RMSNorm gamma buffer
        kv_rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(kv_rmsnorm_gamma_cb, gamma_tensor)
        kv_rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = kv_rmsnorm_tile_descriptor
        kv_rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = kv_rmsnorm_page_size

        # CB X: KV RMSNorm output buffer
        # kv_rmsnorm_output_cb_format = ttnn.CBFormatDescriptor(
        #    buffer_index=kv_rmsnorm_output_cb,
        #    data_format=data_format,
        #    page_size=kv_rmsnorm_page_size,
        #    tile=kv_rmsnorm_tile_descriptor,
        # )
        # kv_rmsnorm_output_cb_descriptor = ttnn.CBDescriptor(
        #    total_size=kv_rmsnorm_num_tiles * kv_rmsnorm_page_size,
        #    core_ranges=gamma_tensor.memory_config().shard_spec.grid,
        #    format_descriptors=[kv_rmsnorm_output_cb_format],
        # )
        # for testing
        kv_rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(kv_rmsnorm_output_cb, output_tensor)
        kv_rmsnorm_output_cb_descriptor.format_descriptors[0].tile = kv_rmsnorm_tile_descriptor
        kv_rmsnorm_output_cb_descriptor.format_descriptors[0].page_size = kv_rmsnorm_page_size

        krope_tile_size = TILE_1x32.get_tile_size(data_format)
        krope_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)

        rope_tile_descriptor = ttnn.TileDescriptor(rope_tile)
        cos_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=cos_cb,
            data_format=data_format,
            page_size=rope_tile_size,
            tile=rope_tile_descriptor,
        )
        cos_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * rope_tile_size,
            core_ranges=krope_core_grid,
            format_descriptors=[cos_cb_format],
        )
        sin_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=sin_cb,
            data_format=data_format,
            page_size=rope_tile_size,
            tile=rope_tile_descriptor,
        )
        sin_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * rope_tile_size,
            core_ranges=krope_core_grid,
            format_descriptors=[sin_cb_format],
        )
        # CB X: Trans_mat (sharded tensor)
        trans_mat_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(trans_mat_cb, trans_mat_tensor)

        # CB X: Rotated input intermediate (not backed by tensor)
        rotated_interm_format = ttnn.CBFormatDescriptor(
            buffer_index=rotated_input_interm_cb,
            data_format=data_format,
            page_size=krope_tile_size,
            tile=krope_tile_descriptor,
        )
        rotated_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * krope_tile_size,
            core_ranges=krope_core_grid,
            format_descriptors=[rotated_interm_format],
        )

        # CB X: Cos intermediate (not backed by tensor)
        cos_interm_format = ttnn.CBFormatDescriptor(
            buffer_index=cos_interm_cb,
            data_format=data_format,
            page_size=krope_tile_size,
            tile=krope_tile_descriptor,
        )
        cos_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * krope_tile_size,
            core_ranges=krope_core_grid,
            format_descriptors=[cos_interm_format],
        )

        # CB X: Sin intermediate (not backed by tensor)
        sin_interm_format = ttnn.CBFormatDescriptor(
            buffer_index=sin_interm_cb,
            data_format=data_format,
            page_size=krope_tile_size,
            tile=krope_tile_descriptor,
        )
        sin_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * krope_tile_size,
            core_ranges=krope_core_grid,
            format_descriptors=[sin_interm_format],
        )

        k_rope_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=k_rope_output_cb,
            data_format=data_format,
            page_size=krope_tile_size,
            tile=krope_tile_descriptor,
        )
        k_rope_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=1 * krope_tile_size,
            core_ranges=krope_core_grid,
            format_descriptors=[k_rope_output_cb_format],
        )
        # ========================================================================
        # Semaphore descriptors
        # ========================================================================

        # Gather semaphores (ID 2 and 3 - two semaphores for NOC0 and NOC1, but only NOC0 is used)
        gather_noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc0_receiver_semaphore_id,
            core_ranges=input_core_grid,
            initial_value=0,
        )

        gather_noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc1_receiver_semaphore_id,
            core_ranges=input_core_grid,
            initial_value=0,
        )

        # ========================================================================
        # Kernel descriptors
        # ========================================================================

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/kv_cache_branch/kernels/kv_cache_branch_kernel.cpp",
            core_ranges=input_core_grid,
            # NCRISC named compile-time args:
            ncrisc_named_compile_time_args=dkv_matmul_ncrisc_named_compile_time_args
            + kv_rmsnorm_ncrisc_named_compile_time_args
            + dkv_gather_sender_named_compile_time_args
            + krope_ncrisc_named_compile_time_args,
            # BRISC named compile-time args
            brisc_named_compile_time_args=dkv_gather_receiver_named_compile_time_args
            + kv_rmsnorm_brisc_named_compile_time_args
            + krope_brisc_named_compile_time_args,
            # BRISC common runtime args: KV cache buffer address and write position
            brisc_common_runtime_args=[
                kv_cache_buffer_addr,
                position_ids_tensor_addr,
            ],
            # TRISC named compile-time args
            trisc_named_compile_time_args=kv_rmsnorm_trisc_named_compile_time_args
            + dkv_matmul_trisc_named_compile_time_args
            + rmsnorm_compute_named_compile_time_args
            + krope_trisc_named_compile_time_args,
            # TRISC common runtime args: epsilon (used by rmsnorm compute)
            trisc_common_runtime_args=[
                epsilon_packed,
                kv_scalar_packed,
            ],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            # Per-core compile-time role differentiation
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_dkv_matmul_core",
                    core_range=dkv_matmul_weights_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_kv_rmsnorm_core",
                    core_range=gamma_tensor.memory_config().shard_spec.grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_knope_core",
                    core_range=dkv_gather_sender_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_krope_core",
                    core_range=krope_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="start_tile_offset",
                    core_values=start_tile_offset_core_values,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                dkv_matmul_input_cb_descriptor,
                dkv_matmul_output_cb_descriptor,
                dkv_matmul_weights_cb_descriptor,
                kv_rmsnorm_input_cb_descriptor,
                kv_rmsnorm_gamma_cb_descriptor,
                kv_rmsnorm_output_cb_descriptor,
                cos_cb_descriptor,
                sin_cb_descriptor,
                trans_mat_cb_descriptor,
                rotated_interm_cb_descriptor,
                cos_interm_cb_descriptor,
                sin_interm_cb_descriptor,
                k_rope_output_cb_descriptor,
            ],
            semaphores=[
                gather_noc0_receiver_semaphore_descriptor,  # ID 2
                gather_noc1_receiver_semaphore_descriptor,  # ID 3
            ],
        )

        # Execute generic op
        # cos_tensor and sin_tensor are accessed by DRAM address, not as io tensors
        io_tensors = [
            input_tensor,
            dkv_matmul_weights_tensor,
            gamma_tensor,
            trans_mat_tensor,
            kv_cache_tensor,
            output_tensor,
        ]
        output = ttnn.generic_op(io_tensors, program_descriptor)
        return output
