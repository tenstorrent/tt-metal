# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post-Decode: Mcast + Matmul fused operation for LM head vocab projection.

Multicasts input_tensor from a single sender core to all cores in the device grid,
then each matmul core computes a local matmul with its vocab weight shard.

- input_tensor (in0): [1, K] on sender core
- vocab_tensor (in1): [K, N_total] width-sharded across matmul cores as [K, N_per_core]
- output: [1, N_total] width-sharded across matmul cores as [1, N_per_core]

CB Layout:
- CB 0: mcast_src (input_tensor on sender core, tensor-backed)
- CB 1: mcast_dst / matmul_in0 (all device grid cores, intermediate)
- CB 2: matmul_in1 (vocab weights on matmul cores, tensor-backed)
- CB 16: matmul_out (output on matmul cores, tensor-backed)
"""

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class PostDecode:
    """
    Post-decode LM head vocab projection: mcast + matmul via ttnn.generic_op.

    Sender core multicasts input_tensor to all device cores, then each matmul
    core computes [1, K] x [K, N_per_core] with its width-sharded vocab weight shard.
    """

    @staticmethod
    def golden(input_tensor, vocab_tensor):
        """
        PyTorch reference implementation of matmul for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [M, K]
            vocab_tensor: Vocab tensor (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N]
        """
        return input_tensor @ vocab_tensor

    @staticmethod
    def op(
        input_tensor,
        vocab_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute mcast + matmul operation using generic_op.

        Multicasts input_tensor from its sender core to all cores in the full device grid,
        then each matmul core computes a local matmul with its vocab weight shard.

        Args:
            input_tensor: Input tensor [1, K] height-sharded in L1 on a single sender core
            vocab_tensor: Vocab weights [K, N_total] width-sharded across matmul cores as [K, N_per_core]
            output_tensor: Pre-allocated output [1, N_total] width-sharded across matmul cores
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with matmul result
        """
        device = input_tensor.device()

        # ====================================================================
        # Shape validation
        # ====================================================================
        a_shape = input_tensor.shape
        b_shape = vocab_tensor.shape
        in0_tile = input_tensor.get_tile()
        in1_tile = vocab_tensor.get_tile()
        assert (
            a_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"M ({a_shape[0]}) must be a single tile with height same as tile_height ({in0_tile.tile_shape[0]})"
        assert (
            a_shape[1] % in0_tile.tile_shape[1] == 0
        ), f"K ({a_shape[1]}) must be divisible by tile_width ({in0_tile.tile_shape[1]})"
        assert a_shape[1] == b_shape[0], f"in0 K ({a_shape[1]}) must equal in1 K ({b_shape[0]})"
        num_tiles_k = a_shape[1] // in0_tile.tile_shape[1]

        logger.info(f"num_tiles_k: {num_tiles_k}")

        out_shape = output_tensor.shape
        out_tile = output_tensor.get_tile()
        assert (
            out_shape[0] // out_tile.tile_shape[0] == 1
        ), f"M ({out_shape[0]}) must be a single tile with height same as tile_height ({out_tile.tile_shape[0]})"

        # ====================================================================
        # Core grid configuration
        # ====================================================================
        # Sender core: from input_tensor (must be single core)
        sender_core_grid = input_tensor.memory_config().shard_spec.grid
        assert sender_core_grid.num_cores() == 1, "input_tensor must be sharded on a single sender core"
        sender_core = list(sender_core_grid.ranges())[0].start

        # Matmul cores: from vocab_tensor (multiple cores with weight shards)
        matmul_core_grid = vocab_tensor.memory_config().shard_spec.grid

        # Mcast grid = full device compute grid (includes sender + all matmul cores)
        device_grid_size = device.compute_with_storage_grid_size()
        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1)
        )

        logger.info(f"Full device grid: {device_grid_size.x - 1}, {device_grid_size.y - 1}")

        mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])
        num_mcast_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Build mcast receiver grid = mcast grid minus sender core
        mcast_receiver_ranges = []
        for row in range(mcast_grid.start.y, mcast_grid.end.y + 1):
            for col in range(mcast_grid.start.x, mcast_grid.end.x + 1):
                if col == sender_core.x and row == sender_core.y:
                    continue
                mcast_receiver_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
        mcast_receiver_grid = ttnn.CoreRangeSet(mcast_receiver_ranges)

        # All cores = mcast grid (sender is already included)
        all_cores = mcast_grid_set

        # Determine if sender is part of the mcast rectangle
        is_part_of_receiver_grid = mcast_grid.contains(sender_core)

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # ====================================================================
        # Per-core output width
        # ====================================================================
        weights_shard_spec = vocab_tensor.memory_config().shard_spec
        n_per_core = weights_shard_spec.shape[1]

        out_w_per_core = n_per_core // out_tile.tile_shape[1]
        logger.info(f"out_w_per_core {out_w_per_core}")

        # ====================================================================
        # Mcast data size
        # ====================================================================
        data_format = input_tensor.dtype
        input_tile_size = in0_tile.get_tile_size(data_format)
        mcast_data_size_bytes = num_tiles_k * input_tile_size

        # ====================================================================
        # CB indices
        # ====================================================================
        mcast_src_cb = 0  # input_tensor on sender core (tensor-backed)
        mcast_dst_cb = 1  # Mcast destination = matmul in0 (all mcast grid cores, intermediate)
        matmul_in1_cb = 2  # vocab_tensor weights on matmul cores (tensor-backed)
        matmul_out_cb = 16  # Matmul output on matmul cores (tensor-backed)

        # ====================================================================
        # Semaphore IDs
        # ====================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # ====================================================================
        # NCRISC compile-time args
        # ====================================================================
        ncrisc_named_compile_time_args = [
            # Mcast source (for setup_sharded_buffer on sender core)
            ("mcast_src_cb", mcast_src_cb),
            ("mcast_src_num_pages", num_tiles_k),
            # Mcast receiver
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_dst_num_pages", num_tiles_k),
            # Matmul
            ("matmul_in0", mcast_dst_cb),  # Matmul reads from mcast destination
            ("matmul_in1", matmul_in1_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", out_w_per_core),
        ]

        # ====================================================================
        # BRISC compile-time args
        # ====================================================================
        brisc_named_compile_time_args = [
            # Mcast sender
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", num_mcast_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", mcast_src_cb),
            ("mcast_src_num_pages", num_tiles_k),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_is_part_of_receiver_grid", is_part_of_receiver_grid),
        ]

        # ====================================================================
        # TRISC compile-time args
        # ====================================================================
        trisc_named_compile_time_args = [
            ("matmul_in0", mcast_dst_cb),  # Matmul reads from mcast destination
            ("matmul_in1", matmul_in1_cb),
            ("matmul_out", matmul_out_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", out_w_per_core),
        ]

        # ====================================================================
        # Circular buffer descriptors
        # ====================================================================
        # CB 0: Mcast source — input_tensor on sender core (tensor-backed, read by BRISC)
        mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(mcast_src_cb, input_tensor)

        # CB 1: Mcast destination — intermediate buffer on all device grid cores
        mcast_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
        mcast_dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=mcast_dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=mcast_dst_tile_descriptor,
        )
        mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_tiles_k * input_tile_size,
            core_ranges=all_cores,
            format_descriptors=[mcast_dst_cb_format],
        )

        # CB 2: Matmul weights — vocab_tensor, tensor-backed on matmul cores
        matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, vocab_tensor)

        # CB 16: Matmul output — tensor-backed on matmul cores
        matmul_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_out_cb, output_tensor)

        # ====================================================================
        # Semaphore descriptors
        # ====================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=mcast_data_sender_semaphore_id,
                core_ranges=all_cores,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=mcast_data_receiver_semaphore_id,
                core_ranges=all_cores,
                initial_value=0,
            ),
        ]

        # ====================================================================
        # Unified kernel descriptor
        # ====================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/post_decode/kernels/post_decode_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_mcast_sender_core",
                    core_range=sender_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_mcast_receiver_core",
                    core_range=mcast_receiver_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # ====================================================================
        # Program descriptor
        # ====================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                mcast_src_cb_descriptor,
                mcast_dst_cb_descriptor,
                matmul_in1_cb_descriptor,
                matmul_out_cb_descriptor,
            ],
            semaphores=semaphore_descriptors,
        )

        # Execute generic op
        # Order must match tensor-backed CBs: CB0(input_tensor), CB2(vocab_tensor/weights), CB16(output)
        io_tensors = [input_tensor, vocab_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
