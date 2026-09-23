# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.vision.vision_block import VisionBlock
from models.demos.gemma4.tt.vision.vision_rotary_embedding import VisionRotaryEmbedding
from models.tt_transformers.tt.common import get_rot_transformation_mat


class VisionTransformer(LightweightModule):
    """
    Gemma-4 vision encoder.

    Runs the on-device transformer stack: 2D rotary cos/sin generation -> transformer blocks.
    Patch embedding is done upstream in ``VisionTower``; patch merging / pooling is done
    downstream. This module operates on already-embedded patch hidden states.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        tt_ccl,
        weight_cache_path,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            mesh_device (ttnn.mesh_device): Mesh device for the model
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Rotary cos/sin generator (patch embedding lives upstream in ``VisionTower``).
        self.rotary_embedding = VisionRotaryEmbedding(
            mesh_device=args.mesh_device,
            args=args,
            dtype=ttnn.bfloat16,
        )

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.num_hidden_layers):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                tt_ccl=tt_ccl,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
            )
            self.blocks.append(block)

    def forward(
        self,
        inputs_embeds,
        pixel_position_ids,
        unpadded_seq_len,
        seq_len,
    ):
        """
        Vision encoder forward: rotary cos/sin -> transformer blocks.

        Args:
            inputs_embeds (ttnn.Tensor): Embedded patch hidden states ``[1, batch, num_patches, hidden_dim]``
                (produced upstream by ``VisionTower``'s patch embedder).
            pixel_position_ids (ttnn.Tensor): Patch (x, y) positions ``[batch, num_patches, 2]``
                (int32, ROW_MAJOR; padding patches are ``(-1, -1)``).
            unpadded_seq_len (int): True number of patches (output is sliced back to this).
            seq_len (int): Padded sequence length the blocks run at.

        Returns:
            ttnn.Tensor: Encoder output ``[1, batch, unpadded_seq_len, hidden_dim]``.
        """
        num_patches = pixel_position_ids.shape[1]

        # Pad the embedded sequence seq -> seq_len. When padding, ``ttnn.pad`` allocates a new
        # tensor, so free the caller's input here; in the no-pad case the first block frees it.
        x = inputs_embeds
        if seq_len > num_patches:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=0.0)
            ttnn.deallocate(inputs_embeds)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Rotary cos/sin (Meta interleaved). The rotary matmul needs a float position tensor, so we
        # apply a functional int->bf16 cast here (the rotary module itself stays typecast-free).
        pos_tile = ttnn.to_layout(pixel_position_ids, ttnn.TILE_LAYOUT)
        pos_bf16 = ttnn.typecast(pos_tile, ttnn.bfloat16)
        ttnn.deallocate(pos_tile)
        cos, sin = self.rotary_embedding(pos_bf16)  # [1, B, num_patches, head_dim]
        ttnn.deallocate(pos_bf16)
        if seq_len > num_patches:
            # Pad the sequence with an identity rotation (cos=1, sin=0) for the padding tokens.
            cos = ttnn.pad(cos, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=1.0)
            sin = ttnn.pad(sin, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=0.0)
        rot_mats = [cos, sin]

        for block in self.blocks:
            x = block(
                x,
                rot_mats=rot_mats,
            )

        x = x[:, :, :unpadded_seq_len, :]
        return x
