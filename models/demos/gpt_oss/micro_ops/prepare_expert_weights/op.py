# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Fused prepare_expert_weights operation.

This implements a fused kernel for transforming routing weights from
[B, 1, S, K] to [K, 1, B*S, H] format with broadcast across hidden dimension.

The operation fuses:
1. reshape [B, 1, S, K] -> [B*S, K]
2. repeat along hidden dimension
3. permute to [K, 1, B*S, H]
4. layout conversion to TILE

This avoids multiple intermediate tensor allocations and memory round-trips.
"""

import math

import torch

import ttnn


class PrepareGptOssExpertsTensorSingleCore:
    """
    Single-core fused implementation for preparing expert routing weights.

    Transforms routing weights from [B, 1, S, K] to [K, 1, B*S, H] format,
    broadcasting each weight value across the hidden dimension.
    """

    @staticmethod
    def golden(topk_expert_weights: torch.Tensor, num_experts_per_tok: int, hidden_size: int) -> torch.Tensor:
        """
        PyTorch reference implementation for validation.

        Args:
            topk_expert_weights: Routing weights [B, 1, S, K] or [B*S, K]
            num_experts_per_tok: Number of experts per token (K)
            hidden_size: Hidden dimension size (H)

        Returns:
            Transformed weights [K, 1, B*S, H]
        """
        # Handle both [B, 1, S, K] and [B*S, K] input shapes
        if topk_expert_weights.dim() == 4:
            B, _, S, K = topk_expert_weights.shape
            batch_seq = B * S
            weights = topk_expert_weights.reshape(batch_seq, K)
        else:
            batch_seq, K = topk_expert_weights.shape
            weights = topk_expert_weights

        # Expand to [B*S, K, H] by broadcasting along last dim
        weights = weights.unsqueeze(-1).expand(-1, -1, hidden_size)

        # Permute to [K, B*S, H]
        weights = weights.permute(1, 0, 2)

        # Add middle dimension: [K, 1, B*S, H]
        weights = weights.unsqueeze(1)

        return weights.contiguous()

    @staticmethod
    def op(
        input_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        num_experts_per_tok: int,
        hidden_size: int,
    ) -> ttnn.Tensor:
        """
        Execute fused prepare_expert_weights using generic_op.

        This kernel reads routing weights and broadcasts each weight across
        the hidden dimension while performing the layout transformation.

        Args:
            input_tensor: Input weights [B*S, K] sharded in L1 (row-major, 1x32 tiles)
            output_tensor: Pre-allocated output [K, 1, B*S, H] sharded in L1 (tile layout)
            num_experts_per_tok: Number of experts per token (K)
            hidden_size: Hidden dimension size (H)

        Returns:
            Output tensor with transformed weights

        Notes:
            - Input must be in ROW_MAJOR layout with 1x32 tiles for efficient scalar access
            - Output must be in TILE layout with 32x32 tiles
            - Both tensors must be sharded on a single core
        """
        # Validate input tensor
        input_shape = input_tensor.shape
        assert len(input_shape) == 2, f"Input must be 2D [B*S, K], got shape {input_shape}"
        batch_seq = input_shape[0]
        K = input_shape[1]
        assert K == num_experts_per_tok, f"Input K dim ({K}) must match num_experts_per_tok ({num_experts_per_tok})"

        # Validate output tensor
        output_shape = output_tensor.shape
        assert len(output_shape) == 4, f"Output must be 4D [K, 1, B*S, H], got shape {output_shape}"
        assert output_shape[0] == K, f"Output dim 0 ({output_shape[0]}) must equal K ({K})"
        assert output_shape[1] == 1, f"Output dim 1 must be 1, got {output_shape[1]}"
        assert output_shape[2] == batch_seq, f"Output dim 2 ({output_shape[2]}) must equal B*S ({batch_seq})"
        assert output_shape[3] == hidden_size, f"Output dim 3 ({output_shape[3]}) must equal H ({hidden_size})"

        # Get tensor properties
        input_dtype = input_tensor.dtype
        output_dtype = output_tensor.dtype
        input_tile = input_tensor.get_tile()
        output_tile = output_tensor.get_tile()

        # Calculate tile dimensions
        input_tile_height, input_tile_width = input_tile.tile_shape
        output_tile_height, output_tile_width = output_tile.tile_shape

        # Calculate sizes
        input_tile_size = input_tile.get_tile_size(input_dtype)
        output_tile_size = output_tile.get_tile_size(output_dtype)

        # Calculate number of input and output tiles
        # Input: [B*S, K] with tiles of shape [input_tile_height, input_tile_width]
        num_input_tiles_bs = math.ceil(batch_seq / input_tile_height)
        num_input_tiles_k = math.ceil(K / input_tile_width)
        num_input_tiles = num_input_tiles_bs * num_input_tiles_k

        # Output: [K, 1, B*S, H] - we process as [K * B*S, H] for tiling
        # Each (k, bs) pair produces H/output_tile_width tiles along the H dimension
        num_output_tiles_h = math.ceil(hidden_size / output_tile_width)
        num_output_tiles_per_weight = num_output_tiles_h  # tiles per (k, bs) weight
        total_output_tiles = K * batch_seq * num_output_tiles_per_weight

        # For simplified single-core: assume B*S fits in one tile height
        # and K fits in one tile width for input
        assert batch_seq <= input_tile_height, f"B*S ({batch_seq}) must fit in input tile height ({input_tile_height})"
        assert K <= input_tile_width, f"K ({K}) must fit in input tile width ({input_tile_width})"

        # Get core grid from tensor memory config (single core)
        all_cores = input_tensor.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, "Only single core is supported"

        # CB indices
        input_cb = 0  # Input weights [B*S, K]
        scalar_cb = 1  # Scalar broadcast buffer
        output_cb = 2  # Output buffer

        # Create tile descriptors
        input_tile_descriptor = ttnn.TileDescriptor(input_tile)
        output_tile_descriptor = ttnn.TileDescriptor(output_tile)

        # CB 0: Input (from sharded tensor)
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        # CB 1: Scalar buffer for broadcasting (single output tile size)
        # This holds one tile that we fill with the scalar value
        scalar_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=scalar_cb,
            data_format=output_dtype,
            page_size=output_tile_size,
            tile=output_tile_descriptor,
        )
        scalar_cb_descriptor = ttnn.CBDescriptor(
            total_size=output_tile_size,
            core_ranges=all_cores,
            format_descriptors=[scalar_cb_format],
        )

        # CB 2: Output (from sharded tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)

        # Reader kernel - reads input weights and signals ready
        reader_compile_time_args = [
            input_cb,
            num_input_tiles,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Compute kernel - broadcasts each weight across hidden dimension
        compute_compile_time_args = [
            input_cb,
            scalar_cb,
            output_cb,
            batch_seq,  # B*S
            num_experts_per_tok,  # K
            hidden_size,  # H
            num_output_tiles_h,  # tiles along H dimension per weight
            input_tile_width,  # for indexing into input tile
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            ),
        )

        # Writer kernel - signals output tiles are ready
        writer_compile_time_args = [
            output_cb,
            total_output_tiles,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[in_cb_descriptor, scalar_cb_descriptor, out_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output


class PrepareGptOssExpertsTensorPipelined:
    """
    Pipelined single-core implementation with optimized tile filling.

    Uses block processing to maximize throughput by:
    1. Processing multiple H tiles per dst register acquire
    2. Using 32-bit writes for faster scalar tile filling
    """

    @staticmethod
    def golden(topk_expert_weights: torch.Tensor, num_experts_per_tok: int, hidden_size: int) -> torch.Tensor:
        """PyTorch reference - same as single core."""
        return PrepareGptOssExpertsTensorSingleCore.golden(
            topk_expert_weights, num_experts_per_tok, hidden_size
        )

    @staticmethod
    def op(
        input_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        num_experts_per_tok: int,
        hidden_size: int,
        tiles_per_block: int = 4,
    ) -> ttnn.Tensor:
        """
        Execute fused prepare_expert_weights with pipelined processing.

        Args:
            input_tensor: Input weights [B*S, K] sharded in L1
            output_tensor: Pre-allocated output [K, 1, B*S, H] sharded in L1
            num_experts_per_tok: Number of experts per token (K)
            hidden_size: Hidden dimension size (H)
            tiles_per_block: Number of tiles to process per dst register acquire

        Returns:
            Output tensor with transformed weights
        """
        # Validate input tensor
        input_shape = input_tensor.shape
        assert len(input_shape) == 2, f"Input must be 2D [B*S, K], got shape {input_shape}"
        batch_seq = input_shape[0]
        K = input_shape[1]
        assert K == num_experts_per_tok, f"Input K dim ({K}) must match num_experts_per_tok ({num_experts_per_tok})"

        # Get tensor properties
        input_dtype = input_tensor.dtype
        output_dtype = output_tensor.dtype
        input_tile = input_tensor.get_tile()
        output_tile = output_tensor.get_tile()

        # Calculate tile dimensions
        input_tile_height, input_tile_width = input_tile.tile_shape
        output_tile_height, output_tile_width = output_tile.tile_shape

        # Calculate sizes
        input_tile_size = input_tile.get_tile_size(input_dtype)
        output_tile_size = output_tile.get_tile_size(output_dtype)

        # Calculate number of tiles
        num_input_tiles_bs = math.ceil(batch_seq / input_tile_height)
        num_input_tiles_k = math.ceil(K / input_tile_width)
        num_input_tiles = num_input_tiles_bs * num_input_tiles_k

        num_output_tiles_h = math.ceil(hidden_size / output_tile_width)
        total_output_tiles = K * batch_seq * num_output_tiles_h

        # Validate constraints
        assert batch_seq <= input_tile_height, f"B*S ({batch_seq}) must fit in input tile height ({input_tile_height})"
        assert K <= input_tile_width, f"K ({K}) must fit in input tile width ({input_tile_width})"

        # Get core grid
        all_cores = input_tensor.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, "Only single core is supported"

        # CB indices
        input_cb = 0
        scalar_cb = 1
        output_cb = 2

        # Create tile descriptors
        input_tile_descriptor = ttnn.TileDescriptor(input_tile)
        output_tile_descriptor = ttnn.TileDescriptor(output_tile)

        # CB descriptors
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        scalar_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=scalar_cb,
            data_format=output_dtype,
            page_size=output_tile_size,
            tile=output_tile_descriptor,
        )
        scalar_cb_descriptor = ttnn.CBDescriptor(
            total_size=output_tile_size,
            core_ranges=all_cores,
            format_descriptors=[scalar_cb_format],
        )

        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)

        # Reader kernel
        reader_compile_time_args = [input_cb, num_input_tiles]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Pipelined compute kernel
        compute_compile_time_args = [
            input_cb,
            scalar_cb,
            output_cb,
            batch_seq,
            num_experts_per_tok,
            hidden_size,
            num_output_tiles_h,
            input_tile_width,
            tiles_per_block,
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/compute_pipelined.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            ),
        )

        # Writer kernel
        writer_compile_time_args = [output_cb, total_output_tiles]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[in_cb_descriptor, scalar_cb_descriptor, out_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output


class PrepareGptOssExpertsTensorMultiCore:
    """
    Multi-core fused implementation for preparing expert routing weights.

    Distributes work across multiple cores where each core handles a subset
    of the K experts. This provides better parallelism for larger K values.
    """

    @staticmethod
    def golden(topk_expert_weights: torch.Tensor, num_experts_per_tok: int, hidden_size: int) -> torch.Tensor:
        """PyTorch reference - same as single core."""
        return PrepareGptOssExpertsTensorSingleCore.golden(
            topk_expert_weights, num_experts_per_tok, hidden_size
        )

    @staticmethod
    def op(
        input_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        num_experts_per_tok: int,
        hidden_size: int,
    ) -> ttnn.Tensor:
        """
        Execute fused prepare_expert_weights on multiple cores.

        Each core processes one or more experts (k values), broadcasting
        weights across the hidden dimension.

        Args:
            input_tensor: Input weights, sharded across cores by expert
            output_tensor: Pre-allocated output, sharded across cores by expert
            num_experts_per_tok: Number of experts per token (K)
            hidden_size: Hidden dimension size (H)

        Returns:
            Output tensor with transformed weights
        """
        # Get tensor properties
        input_dtype = input_tensor.dtype
        output_dtype = output_tensor.dtype
        input_tile = input_tensor.get_tile()
        output_tile = output_tensor.get_tile()

        # Get core grid from tensor memory config
        input_memory_config = input_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        num_cores = input_core_grid.num_cores()
        assert num_cores == output_core_grid.num_cores(), "Input and output must have same number of cores"

        # Calculate per-core work distribution
        # Assume each core handles batch_seq weights for its assigned expert(s)
        input_shard_shape = input_memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape

        shard_batch_seq = input_shard_shape[0]
        shard_k = input_shard_shape[1]  # experts per core

        # Tile calculations
        input_tile_size = input_tile.get_tile_size(input_dtype)
        output_tile_size = output_tile.get_tile_size(output_dtype)
        output_tile_height, output_tile_width = output_tile.tile_shape

        num_output_tiles_h = math.ceil(hidden_size / output_tile_width)
        num_input_tiles = math.ceil(shard_batch_seq * shard_k / (input_tile.tile_shape[0] * input_tile.tile_shape[1]))
        total_output_tiles_per_core = shard_k * shard_batch_seq * num_output_tiles_h

        # CB indices
        input_cb = 0
        scalar_cb = 1
        output_cb = 2

        # Create tile descriptors
        input_tile_descriptor = ttnn.TileDescriptor(input_tile)
        output_tile_descriptor = ttnn.TileDescriptor(output_tile)

        # CB descriptors
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        scalar_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=scalar_cb,
            data_format=output_dtype,
            page_size=output_tile_size,
            tile=output_tile_descriptor,
        )
        scalar_cb_descriptor = ttnn.CBDescriptor(
            total_size=output_tile_size,
            core_ranges=input_core_grid,
            format_descriptors=[scalar_cb_format],
        )

        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)

        # Reader kernel
        reader_compile_time_args = [
            input_cb,
            num_input_tiles,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=input_core_grid,
            compile_time_args=reader_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Compute kernel - same logic but operates on shard
        compute_compile_time_args = [
            input_cb,
            scalar_cb,
            output_cb,
            shard_batch_seq,  # B*S per shard
            shard_k,  # K per shard
            hidden_size,  # H (full)
            num_output_tiles_h,
            input_tile.tile_shape[1],
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=input_core_grid,
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            ),
        )

        # Writer kernel
        writer_compile_time_args = [
            output_cb,
            total_output_tiles_per_core,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/gpt_oss/micro_ops/prepare_expert_weights/kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=input_core_grid,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[in_cb_descriptor, scalar_cb_descriptor, out_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
