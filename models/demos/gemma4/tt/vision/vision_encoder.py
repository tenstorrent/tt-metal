# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.vision.vision_block import VisionBlock
from models.demos.gemma4.tt.vision.vision_patch_embedder import VisionPatchEmbedder
from models.demos.gemma4.tt.vision.vision_rotary_embedding import VisionRotaryEmbedding
from models.tt_transformers.tt.common import get_rot_transformation_mat


class VisionTransformer(LightweightModule):
    """
    Gemma-4 vision tower (encoder).

    Runs the full on-device pipeline: patch embedding -> 2D rotary cos/sin generation ->
    transformer blocks. Patch merging / pooling is done outside this class.
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

        # Patch embedder (input_proj + 2D positional embeddings) and rotary cos/sin generator.
        self.patch_embedder = VisionPatchEmbedder(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=f"{args.get_state_dict_prefix('VisionTransformer')}.patch_embedder.",
            weight_cache_path=weight_cache_path,
            dtype=ttnn.bfloat16,
        )
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
        pixel_values,
        pixel_position_ids,
        padding_positions,
        unpadded_seq_len,
        seq_len,
    ):
        """
        Full vision-tower forward: patch embed -> rotary cos/sin -> transformer blocks.

        Args:
            pixel_values (ttnn.Tensor): Flattened patch pixels ``[1, batch, num_patches, 3*patch_size^2]``.
            pixel_position_ids (ttnn.Tensor): Patch (x, y) positions ``[batch, num_patches, 2]``
                (int32, ROW_MAJOR; padding patches are ``(-1, -1)``).
            padding_positions (ttnn.Tensor): ``[batch, num_patches]`` (uint32/int32, nonzero = padding).
            unpadded_seq_len (int): True number of patches (output is sliced back to this).
            seq_len (int): Padded sequence length the blocks run at.

        Returns:
            ttnn.Tensor: Encoder output ``[1, batch, unpadded_seq_len, hidden_dim]``.
        """
        num_patches = pixel_position_ids.shape[1]

        # Patch embedding (projected patches + 2D positional embeddings), then pad seq -> seq_len.
        x = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)  # [1, B, num_patches, H]
        if seq_len > num_patches:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=0.0)
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
