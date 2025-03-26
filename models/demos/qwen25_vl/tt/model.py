# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
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
        paged_attention_config=None,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            mesh_device (ttnn.mesh_device): Mesh device for the model
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
            paged_attention_config (PagedAttentionConfig, optional): Configuration for paged attention
            num_layers (int, optional): Number of layers to use. If None, use all layers.
        """
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path
        self.paged_attention_config = paged_attention_config
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
                paged_attention_config=paged_attention_config,
            )
            self.blocks.append(block)

    def forward(
        self,
        x,
        cu_seqlens,
        cu_window_seqlens,
        rot_mats,
        user_id=0,
        page_table=None,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths
            cu_window_seqlens (torch.Tensor): Cumulative window sequence lengths
            rot_mats (list): Rotation matrices for positional embeddings
            user_id (int): User ID
            page_table (ttnn.Tensor, optional): Page table for paged attention

        Returns:
            ttnn.Tensor: Output tensor
        """

        # Forward through each block
        hidden_states = x
        for i, block in enumerate(self.blocks):
            # Determine which attention type to use (full or windowed)
            if i in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            # Forward through block
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rot_mats=rot_mats,
                user_id=user_id,
                page_table=page_table,
            )

        return hidden_states
