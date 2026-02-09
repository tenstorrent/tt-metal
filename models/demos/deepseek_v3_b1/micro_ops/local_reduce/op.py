# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Local Reduce micro-op.

Element-wise reduction of N tiles from single CB with optional SiLU activation:
    output = SiLU(in[0] + in[1] + ... + in[n-1])  if apply_silu=True
    output = in[0] + in[1] + ... + in[n-1]        if apply_silu=False

All input tiles come from the same circular buffer.

Face-view optimization: When tile dimensions allow (e.g., 16 [1,32] tiles = 2 [16,16] faces),
the kernel can process fewer, larger tiles for better efficiency.
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# Face dimensions (hardware constant)
FACE_HEIGHT = 16
FACE_WIDTH = 16
FACE_ELEMENTS = FACE_HEIGHT * FACE_WIDTH  # 256


class LocalReduceOp:
    """
    Element-wise reduction of N tiles with optional SiLU activation.

    Computes: output = in[0] + in[1] + ... + in[n-1] (all from same CB)
              or with SiLU: output = SiLU(in[0] + in[1] + ... + in[n-1])

    Supports face-view optimization for small tiles (e.g., [1,32]) that can
    be grouped into [16,16] faces for more efficient processing.
    """

    @staticmethod
    def golden(*inputs, apply_silu=False):
        """PyTorch reference: sum of all inputs with optional SiLU"""
        result = inputs[0]
        for inp in inputs[1:]:
            result = result + inp
        if apply_silu:
            result = torch.nn.functional.silu(result)
        return result

    @staticmethod
    def can_use_face_view(tile_h, tile_w, num_tiles):
        """
        Check if face-view optimization can be applied.

        Face-view optimization requires:
        1. Tiles SMALLER than a face (i.e., not already [16,16])
        2. Small tiles that evenly divide into faces (256 elements)
        3. Total elements form an even number of faces (2, 4, 6, ...)
        """
        elements_per_tile = tile_h * tile_w
        total_elements = elements_per_tile * num_tiles

        # Don't apply optimization if tiles are already face-sized or larger
        if elements_per_tile >= FACE_ELEMENTS:
            return False

        # Check if tiles divide evenly into faces
        if FACE_ELEMENTS % elements_per_tile != 0:
            return False

        # Check if total elements form complete faces
        if total_elements % FACE_ELEMENTS != 0:
            return False

        # Check if we have an even number of faces (for pairwise addition)
        num_faces = total_elements // FACE_ELEMENTS
        if num_faces < 2 or num_faces % 2 != 0:
            return False

        return True

    @staticmethod
    def op(input_tensor, output_tensor, num_tiles, apply_silu=False, use_face_view=None):
        """
        Execute element-wise reduction with optional SiLU using dest register with acc_to_dest.

        Uses single input CB - DST is zeroed at kernel startup, enabling
        direct accumulation with acc_to_dest mode.

        Args:
            input_tensor: Input tensor containing N tiles stacked (shape [N*tile_h, tile_w])
            output_tensor: Pre-allocated output tensor (shape [tile_h, tile_w])
            num_tiles: Number of tiles to accumulate
            apply_silu: Whether to apply SiLU activation after accumulation
            use_face_view: Override auto-detection of face-view optimization.
                           If None, auto-detect based on tile dimensions.

        Returns:
            Output tensor with in[0] + in[1] + ... + in[n-1] (optionally with SiLU)
        """
        all_cores = input_tensor.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, "Only single core is supported"
        assert num_tiles >= 2, "Need at least 2 tiles to accumulate"
        assert num_tiles % 2 == 0, "Number of tiles must be even"

        # Get tile dimensions from input tensor
        input_tile = input_tensor.tile
        tile_h, tile_w = input_tile.tile_shape

        # Determine if face-view optimization applies
        if use_face_view is None:
            use_face_view = LocalReduceOp.can_use_face_view(tile_h, tile_w, num_tiles)

        # CB indices
        in_cb = 0  # Input (N tiles)
        out_cb = 1  # Output (1 tile)

        if use_face_view:
            # Face-view mode: treat input as N [16,16] faces
            elements_per_tile = tile_h * tile_w
            total_elements = elements_per_tile * num_tiles
            kernel_num_tiles = total_elements // FACE_ELEMENTS  # Number of faces
        else:
            # Normal mode: use original tile dimensions
            kernel_num_tiles = num_tiles

        # CB descriptors
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in_cb, input_tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # For face-view mode, override the tile descriptor and page_size
        if use_face_view:
            # Create face tile and get its size
            face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
            face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
            face_tile_size = face_tile.get_tile_size(input_tensor.dtype)

            # Update input CB with face tile configuration
            in_cb_descriptor.format_descriptors[0].tile = face_tile_desc
            in_cb_descriptor.format_descriptors[0].page_size = face_tile_size

            # Update output CB with face tile configuration
            out_cb_descriptor.format_descriptors[0].tile = face_tile_desc
            out_cb_descriptor.format_descriptors[0].page_size = face_tile_size

        # Named compile-time args
        ncrisc_named_compile_time_args = [
            ("local_reduce_in_cb", in_cb),
            ("local_reduce_num_tiles", kernel_num_tiles),
        ]

        trisc_named_compile_time_args = [
            ("local_reduce_in_cb", in_cb),
            ("local_reduce_out_cb", out_cb),
            ("local_reduce_num_tiles", kernel_num_tiles),
            ("local_reduce_apply_silu", 1 if apply_silu else 0),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/local_reduce/kernels/local_reduce_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=apply_silu,  # SiLU uses approximation
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=all_cores,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[in_cb_descriptor, out_cb_descriptor],
        )

        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
