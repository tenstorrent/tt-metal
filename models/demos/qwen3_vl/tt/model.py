# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_vl.reference.functional import qwen3_vision_transformer_preprocess
from models.demos.qwen3_vl.tt.model_config import VisionModelArgs
from models.demos.qwen3_vl.tt.patch_merger import PatchMerger
from models.demos.qwen3_vl.tt.rope import RotarySetup
from models.demos.qwen3_vl.tt.vision_block import VisionBlock
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model import Transformer as TTTransformer
from models.tt_transformers.tt.model_config import TensorGroup


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 3 VL.
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
            state_dict_prefix=args.get_state_dict_prefix("PatchMerger"),
            dtype=dtype,
        )

        self.deepstack_visual_indices = args.hf_config.vision_config.deepstack_visual_indexes
        self.deepstack_merger_list = [
            PatchMerger(
                mesh_device=args.mesh_device,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                state_dict_prefix=args.get_state_dict_prefix("DeepstackMerger", deepstack_merger_num=i),
                dtype=dtype,
                postshuffle_norm=True,
            )
            for i in range(len(self.deepstack_visual_indices))
        ]

    def prepare_input(self, patch_input, seq_len=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            seq_len (int): Sequence length

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        x = patch_input
        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
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
        rot_mats,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths
            rot_mats (list): Rotation matrices for positional embeddings

        Returns:
            ttnn.Tensor: Output tensor
        """
        # Forward through each block
        deepstack_feature_list = []
        for i, block in enumerate(self.blocks):
            # Forward through block
            x = block(
                x,
                rot_mats=rot_mats,
            )
            if i in self.deepstack_visual_indices:
                idx = self.deepstack_visual_indices.index(i)
                deepstack_feature_list.append(self.deepstack_merger_list[idx](x[:, :, :unpadded_seq_len, :]))

        # Merge patches - first remove any sequence length padding
        x = x[:, :, :unpadded_seq_len, :]
        x = self.patch_merger(x)
        return x, deepstack_feature_list


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

        deepstack_indices = [
            idx
            for idx in model_args.hf_config.vision_config.deepstack_visual_indexes
            if idx < model_args.hf_config.vision_config.depth
        ]
        self.deepstack_visual_indexes = deepstack_indices
        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
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

    @property
    def spatial_merge_size(self):
        return self.model_args.hf_config.vision_config.spatial_merge_size

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
        # process pixel_values for each image/video separately
        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []
        deepstack_visual_embeds_list = [None] * len(self.deepstack_visual_indexes)
        # todo)) refactor this code to leverage tt-mesh's ttnn.ShardTensorToMesh(mesh_device, dim=batch_size_dim) for data parallelism
        for grid_thw in all_grid_thw:
            # --- pick out the pixel_values for this users' images (grid_thw.prod() pixels) ---
            pixel_values = all_pixel_values[: grid_thw.prod(), :]
            all_pixel_values = all_pixel_values[grid_thw.prod() :, :]
            # --- Preprocessing ---
            # 1. Calculate total unpadded sequence length
            grid_thw = grid_thw.unsqueeze(0)
            unpadded_seq_len = (grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
            # Calculate padded sequence length (divisible by 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048

            # 2. Use preprocessing function from reference/functional to get indices and embeddings
            cu_seqlens, position_embeddings = qwen3_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw,
                head_dim=self.model_args.head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
            )

            # 3. Use reference model's patch embedding
            patch_input = self.reference_model.patch_embed(pixel_values)
            pos_embeds = self.reference_model.fast_pos_embed_interpolate(grid_thw)
            patch_input = patch_input + pos_embeds

            # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
            # pad sequence length with cos = 1, sin = 0 (identity rotation)
            cos_padded = (
                torch.nn.functional.pad(cos_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sin_padded = (
                torch.nn.functional.pad(sin_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            # Convert to TT tensors on the mesh device
            cos = ttnn.from_torch(
                cos_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                # mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
                # todo)) refactor this code to make the intent clear, which is data parallelism
                mesh_mapper=ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=0),
            )
            sin = ttnn.from_torch(
                sin_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                # mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
                # todo)) refactor this code to make the intent clear, which is data parallelism
                mesh_mapper=ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=0),
            )
            rot_mats = [cos, sin]

            # 5. Prepare input tensor for the TT model using window_index
            tt_input = self.tt_model.prepare_input(patch_input, seq_len)

            # --- TT Model Execution ---
            tt_out, deepstack_visual_embeds = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
            )

            # deallocate device tensors that are not needed by decode
            ttnn.deallocate(tt_input)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            # --- Postprocessing ---
            # 1. Convert TT output back to torch tensor
            tt_output_torch = ttnn.to_torch(
                tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=1)
            )

            deepstack_visual_embeds_torch_list = [
                ttnn.to_torch(
                    deepstack_visual_embeds[i],
                    mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=1),
                )
                for i in range(len(deepstack_visual_embeds))
            ]

            # deallocate TT output
            ttnn.deallocate(tt_out)
            [ttnn.deallocate(deepstack_visual_embeds[i]) for i in range(len(deepstack_visual_embeds))]

            # 2. Extract the relevant output part and adjust shape (matching test logic)
            out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
            # Output shape from TT is [1, B=1, S, H_out_padded], slice H and squeeze B, batch dims
            final_output = tt_output_torch[:, 0:1, :, :out_hidden_size].squeeze(0).squeeze(0)
            deepstack_visual_embeds_torch = [
                deepstack_visual_embeds_torch_list[i][:, 0:1, :, :out_hidden_size].squeeze(0).squeeze(0)
                for i in range(len(deepstack_visual_embeds_torch_list))
            ]

            if self.debug:
                logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
                reference_output, deepstack_ref = self.reference_model.forward(pixel_values, grid_thw)
                _, pcc = comp_pcc(reference_output, final_output)
                logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")

            final_outputs.append(final_output)
            for i in range(len(deepstack_visual_embeds_list)):
                if deepstack_visual_embeds_list[i] is None:
                    deepstack_visual_embeds_list[i] = deepstack_visual_embeds_torch[i]
                else:
                    deepstack_visual_embeds_list[i] = torch.cat(
                        [deepstack_visual_embeds_list[i], deepstack_visual_embeds_torch[i]], dim=0
                    )
        # concatenate all the outputs
        return torch.cat(final_outputs, dim=0), deepstack_visual_embeds_list


class Transformer(TTTransformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        # Call parent constructor with vision-specific classes
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            rope_setup_class=RotarySetup,
        )

    def _prepare_cos_sin(self, rot_mats):
        cos_matrix = rot_mats[0]
        sin_matrix = rot_mats[1]
        assert cos_matrix.shape[0] == sin_matrix.shape[0], "cos_matrix and sin_matrix must have the same batch size"
        outputs = []
        for mat in (cos_matrix, sin_matrix):
            outputs.append(
                ttnn.from_torch(
                    # [INFO] Qwen2.5 VL produces cos and sin matrices with shape [batch_size, 1, seq_len, head_dim]
                    mat.expand(cos_matrix.shape[0], -1, -1, -1),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.rope_setup.datatype,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                ),
            )
        return outputs

    def prepare_inputs_prefill(
        self,
        tokens,
        rot_mats,
        deepstack_visual_embeds=None,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
    ):
        assert isinstance(rot_mats[0], torch.Tensor)
        assert isinstance(rot_mats[1], torch.Tensor)
        # tokens is actually embeddings
        assert tokens.dim() == 3, "tokens should be a 3D tensor"  # [batch_size = 1, seq_len, head_dim]
        S = tokens.shape[-2]
        tokens_embd = ttnn.from_torch(
            tokens.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            ),
        )

        if deepstack_visual_embeds is not None:
            deepstack_visual_embeds = [
                ttnn.from_torch(
                    deepstack_visual_embeds[i].unsqueeze(0).unsqueeze(0),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
                    ),
                )
                for i in range(len(deepstack_visual_embeds))
            ]

        # Slice the rot mats to the prefill seqlen
        cos_matrix, sin_matrix = self._prepare_cos_sin(rot_mats=rot_mats)
        assert (
            cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {cos_matrix.shape[2]}"
        tt_rot_mats_prefill = [
            cos_matrix[:, :, start_pos : start_pos + S, :],
            sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table, deepstack_visual_embeds

    def deepstack_process(self, x, deepstack_visual_embeds):
        x = ttnn.add(x, deepstack_visual_embeds)
        return x

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        deepstack_visual_embeds=None,
    ):
        return self.forward(
            x,
            current_pos=None,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            mode=Mode.PREFILL,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    ):
        for i, layer in enumerate(self.layers):
            # No-op if callers already provide the right memory config
            activation_dtype = self.decoders_optimizations.get_tensor_dtype(decoder_id=i, tensor=TensorGroup.ACTIVATION)
            if mode == Mode.DECODE and not self.args.is_galaxy:
                x = ttnn.to_memory_config(x, self.args.get_residual_mem_config(mode, None), activation_dtype)
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )
            if deepstack_visual_embeds is not None and i in range(len(deepstack_visual_embeds)):
                x = self.deepstack_process(x, deepstack_visual_embeds[i])

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        print(mode)
        x = self.norm(x, mode=mode)

        if mode == Mode.PREFILL and self.args.get_lm_head_input_mem_config(mode, None).is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.args.get_lm_head_input_mem_config(mode, None))

        x = self.lm_head(x)

        if mode == Mode.PREFILL:
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x
