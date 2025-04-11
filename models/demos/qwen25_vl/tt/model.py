# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
from models.utility_functions import comp_pcc
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen25_vl.tt.vision_block import VisionBlock
from models.demos.qwen25_vl.tt.patch_merger import PatchMerger
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    standardize_hf_keys,
    convert_rope_style_hf_to_meta,
)


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
        state_dict,
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
        self.fullatt_block_indexes = args.hf_config.vision_config.fullatt_block_indexes

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(args.head_dim)
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

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
            )
            self.blocks.append(block)

        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
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
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
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


class DropInVisionTransformer(torch.nn.Module):
    """Wraps VisionTransformer to be a drop-in replacement for
    Qwen2_5_VisionTransformerPretrainedModel. It uses the reference model
    for certain preprocessing steps like patch embedding and index calculation.
    """

    def __init__(
        self,
        reference_model,
        model_args: VisionModelArgs,
        dtype=ttnn.bfloat8_b,
        debug=False,
    ):
        """
        Initialize the TorchVisionTransformer wrapper.

        Args:
            tt_model (VisionTransformer): Initialized TT VisionTransformer instance.
            reference_model (Qwen2_5_VisionTransformerPretrainedModel): Initialized reference HF model instance.
            model_args (VisionModelArgs): Model configuration arguments.
            mesh_device (ttnn.MeshDevice): The mesh device used by the TT model.
        """
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        # FIXME: state_dict = model_args.load_state_dict()
        state_dict = standardize_hf_keys(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
        state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        # Initialize TT model
        self.tt_model = VisionTransformer(
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

    @property
    def dtype(self):
        return self.reference_model.dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mimicking the Qwen2_5_VisionTransformerPretrainedModel interface.

        Args:
            pixel_values (torch.Tensor): Input pixel values tensor (equivalent to hidden_states for the ref model start).
                                         Shape typically [num_patches, hidden_size] or similar before patch_embed.
            grid_thw (torch.Tensor): Tensor describing the grid dimensions (time, height, width) for each image/video.
                                     Shape [num_images_or_videos, 3].

        Returns:
            torch.Tensor: Output tensor matching the reference model's output shape [total_seq_len, out_hidden_size].
        """
        # --- Preprocessing ---
        # 1. Calculate total unpadded sequence length
        unpadded_seq_len = (grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
        # Calculate padded sequence length (divisible by 128)
        seq_len = ((unpadded_seq_len // 128) + 1) * 128

        # 2. Use preprocessing function from reference/functional to get indices and embeddings
        cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
            seq_len=unpadded_seq_len,
            grid_thw=grid_thw,
            head_dim=self.model_args.head_dim,
            spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
            window_size=self.model_args.hf_config.vision_config.window_size,
            patch_size=self.model_args.hf_config.vision_config.patch_size,
        )

        # Ensure cu_seqlens and cu_window_seqlens are tensors on the correct device
        cu_seqlens = cu_seqlens.to(pixel_values.device)
        cu_window_seqlens = cu_window_seqlens.to(pixel_values.device)

        # 3. Use reference model's patch embedding
        patch_input = self.reference_model.patch_embed(pixel_values)

        # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
        cos_orig, sin_orig = position_embeddings
        cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
        # pad sequence length with cos = 1, sin = 0 (identity rotation)
        cos_padded = (
            torch.nn.functional.pad(cos_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=1).unsqueeze(0).unsqueeze(0)
        )
        sin_padded = (
            torch.nn.functional.pad(sin_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=0).unsqueeze(0).unsqueeze(0)
        )
        # Convert to TT tensors on the mesh device
        cos = ttnn.from_torch(
            cos_padded,
            dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
            layout=ttnn.TILE_LAYOUT,
            device=self.model_args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
        )
        sin = ttnn.from_torch(
            sin_padded,
            dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
            layout=ttnn.TILE_LAYOUT,
            device=self.model_args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
        )
        rot_mats = [cos, sin]

        # 5. Prepare input tensor for the TT model using window_index
        tt_input = self.tt_model.prepare_input(patch_input, window_index)

        # --- TT Model Execution ---
        tt_out = self.tt_model(
            tt_input,
            unpadded_seq_len=unpadded_seq_len,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens,
            rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
        )

        # --- Postprocessing ---
        # 1. Convert TT output back to torch tensor
        mesh_composer = ttnn.ConcatMesh2dToTensor(
            self.model_args.mesh_device, dims=(1, 3), mesh_shape=self.model_args.cluster_shape
        )
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)

        # 2. Extract the relevant output part and adjust shape (matching test logic)
        out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
        # Output shape from TT is [1, B=1, S, H_out_padded], slice H and squeeze B, batch dims
        tt_output_torch = tt_output_torch[:, 0:1, :, :out_hidden_size].squeeze(0).squeeze(0)

        # 3. Apply reverse window indexing to match reference model output order
        reverse_indices = torch.argsort(window_index)
        final_output = tt_output_torch[reverse_indices, :]

        if self.debug:
            logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
            reference_output = self.reference_model.forward(pixel_values, grid_thw)
            _, pcc = comp_pcc(reference_output, final_output)
            logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")
        return final_output
