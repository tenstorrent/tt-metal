# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-core RoPE (Rotary Position Embedding) operation.

This implements rotary position embedding on a single core where:
- Input: [1, batch, n_heads, head_dim] in L1 (HEIGHT_SHARDED)
- Cos: [1, batch, 1, head_dim] in L1 (HEIGHT_SHARDED)
- Sin: [1, batch, 1, head_dim] in L1 (HEIGHT_SHARDED)
- Trans_mat: [1, 1, 1, TILE_SIZE x TILE_SIZE] in L1 (HEIGHT_SHARDED)
- Output: [1, batch, n_heads, head_dim] in L1 (HEIGHT_SHARDED)

The computation is:
    output = (input * cos) + (rotate_half(input) * sin)

where rotate_half is implemented via matrix multiplication with trans_mat.
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def rotate_half_meta_style(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input using Meta-style interleaving.

    For input [a0, a1, a2, a3, ...], produces [-a1, a0, -a3, a2, ...]

    Args:
        x: Input tensor with last dimension being head_dim

    Returns:
        Rotated tensor with same shape
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RopeSingleCore:
    """
    Single-core RoPE implementation using ttnn.generic_op.

    This class implements rotary position embedding as a static operation for single-core execution.
    Uses the Meta-style interleaved format for rotation.
    """

    @staticmethod
    def golden(
        input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        PyTorch reference implementation of RoPE for validation.

        Args:
            input_tensor: Input tensor [batch, n_heads, seq_len, head_dim] or [1, batch, n_heads, head_dim]
            cos: Cosine tensor [max_seq_len, head_dim]
            sin: Sine tensor [max_seq_len, head_dim]
            position_ids: Position indices [batch, seq_len] or [batch]

        Returns:
            Output tensor with RoPE applied
        """
        # Index into cos/sin using position_ids
        cos_selected = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin_selected = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]

        # Apply rotary embedding with Meta-style rotation
        output = (input_tensor * cos_selected) + (rotate_half_meta_style(input_tensor) * sin_selected)
        return output

    @staticmethod
    def op(
        input_tensor,
        cos_tensor,
        sin_tensor,
        trans_mat_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute RoPE operation using generic_op.

        Args:
            input_tensor: Input tensor [1, batch, n_heads, head_dim] (must be HEIGHT_SHARDED)
            cos_tensor: Cosine tensor [1, batch, 1, head_dim] (must be HEIGHT_SHARDED)
            sin_tensor: Sine tensor [1, batch, 1, head_dim] (must be HEIGHT_SHARDED)
            trans_mat_tensor: Transformation matrix for rotate_half (must be HEIGHT_SHARDED)
            output_tensor: Pre-allocated output tensor (must be HEIGHT_SHARDED)
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with RoPE applied
        """
        # Get tensor properties
        data_format = input_tensor.dtype

        # Get shard spec from input tensor
        shard_spec = input_tensor.memory_config().shard_spec
        shard_shape = shard_spec.shape

        # Calculate dimensions in tiles
        # With tiny tiles: shard_shape[0] = n_heads, shard_shape[1] = head_dim
        head_dim_t = shard_shape[1] // ttnn.TILE_SIZE  # head_dim in tiles (Wt)

        # Get core grid from shard spec
        core_grid = shard_spec.grid

        # Calculate tile sizes
        # For tiny tiles, the tile height matches the shard height (num_q_heads_per_core)
        # This is derived from the input tensor's shard shape
        num_q_heads_per_core = shard_shape[0]  # Get from input tensor's shard height
        tile = ttnn.Tile((num_q_heads_per_core, ttnn.TILE_SIZE))
        tile_size = tile.get_tile_size(data_format)

        # Number of tiles for intermediate buffers
        num_interm_tiles = head_dim_t  # Intermediate buffers sized for one head row

        # CB indices (matching C++ implementation)
        input_cb = 0  # c_0
        cos_cb = 1  # c_1
        sin_cb = 2  # c_2
        trans_mat_cb = 3  # c_3
        output_cb = 16  # c_16 (output operands start at 16)
        rotated_input_interm_cb = 24  # c_24
        cos_interm_cb = 25  # c_25
        sin_interm_cb = 26  # c_26

        # Create tile descriptor
        tile_descriptor = ttnn.TileDescriptor(tile)

        # ========================================================================
        # Circular Buffer Descriptors
        # ========================================================================

        # CB 0: Input (sharded tensor)
        input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        # CB 1: Cos (sharded tensor)
        cos_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cos_cb, cos_tensor)

        # CB 2: Sin (sharded tensor)
        sin_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(sin_cb, sin_tensor)

        # CB 3: Trans_mat (sharded tensor)
        trans_mat_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(trans_mat_cb, trans_mat_tensor)

        # CB 16: Output (sharded tensor)
        output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)

        # CB 24: Rotated input intermediate (not backed by tensor)
        rotated_interm_format = ttnn.CBFormatDescriptor(
            buffer_index=rotated_input_interm_cb,
            data_format=data_format,
            page_size=tile_size,
            tile=tile_descriptor,
        )
        rotated_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_interm_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[rotated_interm_format],
        )

        # CB 25: Cos intermediate (not backed by tensor)
        cos_interm_format = ttnn.CBFormatDescriptor(
            buffer_index=cos_interm_cb,
            data_format=data_format,
            page_size=tile_size,
            tile=tile_descriptor,
        )
        cos_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_interm_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[cos_interm_format],
        )

        # CB 26: Sin intermediate (not backed by tensor)
        sin_interm_format = ttnn.CBFormatDescriptor(
            buffer_index=sin_interm_cb,
            data_format=data_format,
            page_size=tile_size,
            tile=tile_descriptor,
        )
        sin_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_interm_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[sin_interm_format],
        )

        # ========================================================================
        # Unified Kernel Descriptor (handles NCRISC, BRISC, TRISC)
        # ========================================================================

        # Named compile-time args for NCRISC
        ncrisc_named_compile_time_args = [
            ("in_cb", input_cb),
            ("cos_cb", cos_cb),
            ("sin_cb", sin_cb),
            ("trans_mat_cb", trans_mat_cb),
            ("Wt", head_dim_t),
        ]

        # Named compile-time args for BRISC (empty - no-op)
        brisc_named_compile_time_args = []

        # Named compile-time args for TRISC
        trisc_named_compile_time_args = [
            ("in_cb", input_cb),
            ("cos_cb", cos_cb),
            ("sin_cb", sin_cb),
            ("trans_mat_cb", trans_mat_cb),
            ("rotated_in_interm_cb", rotated_input_interm_cb),
            ("cos_interm_cb", cos_interm_cb),
            ("sin_interm_cb", sin_interm_cb),
            ("out_cb", output_cb),
            ("Wt", head_dim_t),
        ]

        # Unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/rope/kernels/rope_kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # ========================================================================
        # Program Descriptor
        # ========================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                input_cb_descriptor,
                cos_cb_descriptor,
                sin_cb_descriptor,
                trans_mat_cb_descriptor,
                output_cb_descriptor,
                rotated_interm_cb_descriptor,
                cos_interm_cb_descriptor,
                sin_interm_cb_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, cos_tensor, sin_tensor, trans_mat_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
