# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen25_vl.tt.vision_block import VisionBlock
from models.demos.qwen25_vl.tt.patch_merger import PatchMerger


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 2.5 VL.
    This implements only the transformer blocks part of the vision transformer.
    Patch embedding and merging should be done outside this class.
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        transformation_mats,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            mesh_device (ttnn.mesh_device): Mesh device for the model
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
            num_layers (int, optional): Number of layers to use. If None, use all layers.
        """
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path
        self.fullatt_block_indexes = args.hf_config.vision_config.fullatt_block_indexes

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=transformation_mats,  # Will be provided during forward pass
                args=args,
            )
            self.blocks.append(block)

        self.patch_merger = PatchMerger(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def prepare_input(self, patch_input, window_index):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            window_index (torch.Tensor): Window index tensor

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        spatial_merge_unit = self.args.hf_config.vision_config.spatial_merge_size**2
        x = patch_input.reshape(patch_seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(patch_seq_len, -1)
        seq_len = ((patch_seq_len // 128) + 1) * 128
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len))
        x = self.args.prepare_residual_tensor_prefill(
            x,
            force_replicated=False if self.args.is_galaxy else True,
        )
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        cu_seqlens,
        cu_window_seqlens,
        rot_mats,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths
            cu_window_seqlens (torch.Tensor): Cumulative window sequence lengths
            rot_mats (list): Rotation matrices for positional embeddings

        Returns:
            ttnn.Tensor: Output tensor
        """

        # Forward through each block
        for i, block in enumerate(self.blocks):
            # Determine which attention type to use (full or windowed)
            if i in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            # Forward through block
            x = block(
                x,
                cu_seqlens=cu_seqlens_now,
                rot_mats=rot_mats,
            )

        # Merge patches - first remove any sequence length padding
        x = x[:, :, :unpadded_seq_len, :]
        x = self.patch_merger(x)
        return x
