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
        x = self.args.prepare_residual_tensor_prefill(x)
        return x

    def _slice_unpadded(self, x, unpadded_seq_len, padded_seq_len_list=None):
        """Slice x along dim 2 to remove padding.

        For a list of unpadded lengths (batched prefill), each image occupies
        padded_seq_len_list[i] slots in dim 2; we extract the first
        unpadded_seq_len[i] tokens from each and concatenate them.
        For a scalar unpadded_seq_len, a simple prefix slice is performed.
        """
        if isinstance(unpadded_seq_len, (list, tuple)):
            chunks = []
            offset = 0
            for s, ps in zip(unpadded_seq_len, padded_seq_len_list):
                chunks.append(x[:, :, offset : offset + s, :])
                offset += ps
            return ttnn.concat(chunks, dim=2)
        return x[:, :, :unpadded_seq_len, :]

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
        batch_size=1,
        user_id_tensor=None,
        padded_seq_len_list=None,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim]
            unpadded_seq_len (int | list[int]): Actual (unpadded) sequence length(s).
                Pass a list for batched prefill; a scalar otherwise.
            rot_mats (list): Rotation matrices for positional embeddings
            padded_seq_len_list (list[int] | None): Per-image padded slot counts,
                required when unpadded_seq_len is a list.

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
                batch_size=batch_size,
                user_id_tensor=user_id_tensor,
            )
            if i in self.deepstack_visual_indices:
                idx = self.deepstack_visual_indices.index(i)
                x_unpadded = self._slice_unpadded(x, unpadded_seq_len, padded_seq_len_list)
                deepstack_feature_list.append(self.deepstack_merger_list[idx](x_unpadded))

        # Merge patches - first remove any sequence length padding
        x = self._slice_unpadded(x, unpadded_seq_len, padded_seq_len_list)
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

    def _vision_embed_rows_to_torch_dealloc(self, tt_rows: ttnn.Tensor) -> torch.Tensor:
        """Read row-major vision embeddings to host and free device tensor (avoids ttnn.concat OOM)."""
        mesh = self.model_args.mesh_device
        n = int(tt_rows.shape[0])
        t = ttnn.to_torch(
            tt_rows,
            dtype=torch.bfloat16,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )
        ttnn.deallocate(tt_rows)
        return t[:n, :].contiguous()

    def _vision_seq_lens_and_padded_slots(self, grid_thw: torch.Tensor):
        """Per-image unpadded seq lengths and uniform padded slot count (max over images)."""
        seq_len_list = []
        for i in range(grid_thw.shape[0]):
            g = grid_thw[i : i + 1]
            s = int((g[:, 1] * g[:, 2]).sum().item())
            seq_len_list.append(s)
        max_pad = max((((s // 2048) + 1) * 2048) for s in seq_len_list)
        padded_seq_len_list = [max_pad] * len(seq_len_list)
        return seq_len_list, padded_seq_len_list

    def _vision_rope_cos_sin_batched(self, grid_thw: torch.Tensor, seq_len_list: list, max_pad: int):
        """Build [1, 1, N * max_pad, head_dim] cos/sin for batched vision prefill."""
        cos_chunks, sin_chunks = [], []
        spatial_merge = self.model_args.hf_config.vision_config.spatial_merge_size
        for i, s in enumerate(seq_len_list):
            _, position_embeddings = qwen3_vision_transformer_preprocess(
                seq_len=s,
                grid_thw=grid_thw[i : i + 1],
                head_dim=self.model_args.head_dim,
                spatial_merge_size=spatial_merge,
            )
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
            cos_chunks.append(
                torch.nn.functional.pad(cos_orig, (0, 0, 0, max_pad - s), value=1).unsqueeze(0).unsqueeze(0)
            )
            sin_chunks.append(
                torch.nn.functional.pad(sin_orig, (0, 0, 0, max_pad - s), value=0).unsqueeze(0).unsqueeze(0)
            )
        cos_cat = torch.cat(cos_chunks, dim=2)
        sin_cat = torch.cat(sin_chunks, dim=2)
        return cos_cat, sin_cat

    def _forward_vision_batched_prefill_chunk(self, pixel_values: torch.Tensor, grid_full: torch.Tensor):
        """One TT vision forward for a contiguous subset of images (batched prefill within the chunk)."""
        batch_size = int(grid_full.shape[0])
        assert batch_size >= 1

        seq_len_list, padded_seq_len_list = self._vision_seq_lens_and_padded_slots(grid_full)
        max_pad = padded_seq_len_list[0]

        patch_input = self.reference_model.patch_embed(pixel_values)
        pos_embeds = self.reference_model.fast_pos_embed_interpolate(grid_full)
        patch_input = patch_input + pos_embeds

        padded_chunks = []
        offset = 0
        for s in seq_len_list:
            chunk = patch_input[offset : offset + s, :]
            offset += s
            padded_chunks.append(torch.nn.functional.pad(chunk, (0, 0, 0, max_pad - s)))
        assert offset == patch_input.shape[0], "patch rows must match sum of per-image seq lengths"
        patch_input_batched = torch.cat(padded_chunks, dim=0)
        total_padded_seq_len = sum(padded_seq_len_list)

        cos_padded, sin_padded = self._vision_rope_cos_sin_batched(grid_full, seq_len_list, max_pad)
        cos = ttnn.from_torch(
            cos_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.model_args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
        )
        sin = ttnn.from_torch(
            sin_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.model_args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
        )
        rot_mats = [cos, sin]

        tt_input = self.tt_model.prepare_input(patch_input_batched, total_padded_seq_len)

        tt_out, deepstack_visual_embeds = self.tt_model(
            tt_input,
            unpadded_seq_len=seq_len_list,
            rot_mats=rot_mats,
            batch_size=batch_size,
            padded_seq_len_list=padded_seq_len_list,
        )

        ttnn.deallocate(tt_input)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(rot_mats[0])
        ttnn.deallocate(rot_mats[1])

        out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
        # Reshape/slice may alias tt_out / deepstack parents; cloning before dealloc keeps a valid device buffer
        # for to_torch (host merge path) and avoids "Buffer must be allocated on device".
        final_output_ttnn = ttnn.clone(ttnn.reshape(tt_out[:, 0:1, :, :out_hidden_size], (-1, out_hidden_size)))
        ttnn.deallocate(tt_out)
        deepstack_visual_embeds_list = []
        for i in range(len(deepstack_visual_embeds)):
            parent = deepstack_visual_embeds[i]
            deepstack_visual_embeds_list.append(
                ttnn.clone(ttnn.reshape(parent[:, 0:1, :, :out_hidden_size], (-1, out_hidden_size)))
            )
            ttnn.deallocate(parent)
        return final_output_ttnn, deepstack_visual_embeds_list

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mimicking the Qwen2_5_VisionTransformerPretrainedModel interface.

        Args:
            pixel_values (torch.Tensor): Input pixel values tensor (equivalent to hidden_states for the ref model start).
                                         Shape typically [num_patches, hidden_size] or similar before patch_embed.
            grid_thw (torch.Tensor): Tensor describing the grid dimensions (time, height, width) for each image/video.
                                     Shape [num_images_or_videos, 3].

        Returns:
            (image_out, deepstack): For a single TT vision chunk, both are ttnn. When ``vision_prefill_chunk_size``
            forces multiple chunks, returns **torch** tensors on CPU (correct values) and ``merge_vision_tokens_ttnn``
            uploads after merge — avoids a mismatched device mesh round-trip that corrupts embeddings.
        """
        grid_full = grid_thw
        if grid_full.dim() == 1:
            grid_full = grid_full.unsqueeze(0)
        batch_size = int(grid_full.shape[0])
        assert batch_size >= 1

        chunk_cap = getattr(self.model_args, "vision_prefill_chunk_size", 1)
        if chunk_cap is None or chunk_cap <= 0:
            chunk_cap = batch_size
        else:
            chunk_cap = min(int(chunk_cap), batch_size)

        n_pix_edges = [0]
        for i in range(batch_size):
            n_pix_edges.append(n_pix_edges[-1] + int(torch.prod(grid_full[i]).item()))

        num_iters = (batch_size + chunk_cap - 1) // chunk_cap
        if num_iters == 1:
            gchunk = grid_full
            pvchunk = pixel_values[n_pix_edges[0] : n_pix_edges[batch_size], :]
            final_output_ttnn, deepstack_visual_embeds_list = self._forward_vision_batched_prefill_chunk(
                pvchunk, gchunk
            )
        else:
            # Host-side cat: device concat would allocate another full buffer and OOM.
            main_cpu_chunks = []
            ds_cpu_by_layer = None
            for start in range(0, batch_size, chunk_cap):
                end = min(start + chunk_cap, batch_size)
                gchunk = grid_full[start:end]
                pvchunk = pixel_values[n_pix_edges[start] : n_pix_edges[end], :]
                out_ttnn, ds_list = self._forward_vision_batched_prefill_chunk(pvchunk, gchunk)
                main_cpu_chunks.append(self._vision_embed_rows_to_torch_dealloc(out_ttnn))
                if ds_cpu_by_layer is None:
                    ds_cpu_by_layer = [[self._vision_embed_rows_to_torch_dealloc(d)] for d in ds_list]
                else:
                    for i, d in enumerate(ds_list):
                        ds_cpu_by_layer[i].append(self._vision_embed_rows_to_torch_dealloc(d))
            merged_main = torch.cat(main_cpu_chunks, dim=0)
            final_output_ttnn = merged_main
            deepstack_visual_embeds_list = [torch.cat(parts, dim=0) for parts in ds_cpu_by_layer]
        if self.debug:
            logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
            reference_output = torch.load("models/demos/qwen3_vl/tt/ref_out_visual.pt")
            _cmp = final_output_ttnn
            if not isinstance(_cmp, torch.Tensor):
                _cmp = ttnn.to_torch(
                    _cmp,
                    dtype=torch.bfloat16,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=0),
                )[: int(final_output_ttnn.shape[0]), :]
            _, pcc = comp_pcc(reference_output, _cmp)
            logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")
            import sys

            sys.exit(pcc)

        # concatenate all the outputs
        return final_output_ttnn, deepstack_visual_embeds_list


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
        batch_size=1,
    ):
        assert isinstance(rot_mats[0], torch.Tensor)
        assert isinstance(rot_mats[1], torch.Tensor)
        is_torch_tokens = isinstance(tokens, torch.Tensor)
        assert (tokens.dim() if is_torch_tokens else len(tokens.shape)) == 3
        B, S, H = int(tokens.shape[0]), int(tokens.shape[1]), int(tokens.shape[2])
        if batch_size > 1:
            tokens = tokens.reshape(1, 1, B * S, H) if is_torch_tokens else ttnn.reshape(tokens, (1, 1, B * S, H))
        else:
            tokens = tokens.unsqueeze(1) if is_torch_tokens else ttnn.unsqueeze(tokens, 1)

        if is_torch_tokens:
            tokens_embd = ttnn.from_torch(
                tokens,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
                ),
            )
        else:
            tokens_embd = tokens

        if deepstack_visual_embeds is not None:
            mesh_mapper = ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            )
            reshaped = []
            for ds in deepstack_visual_embeds:
                if isinstance(ds, torch.Tensor):
                    if ds.dim() == 2:
                        x = ds.unsqueeze(0).unsqueeze(0)
                    else:
                        x = ds.reshape(1, 1, -1, ds.shape[-1])
                    reshaped.append(
                        ttnn.from_torch(
                            x,
                            device=self.mesh_device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                    )
                else:
                    if len(ds.shape) == 2:
                        x = ttnn.unsqueeze(ttnn.unsqueeze(ds, 0), 0)
                    else:
                        b_, s_, h_ = int(ds.shape[0]), int(ds.shape[1]), int(ds.shape[2])
                        x = ttnn.reshape(ds, (1, 1, b_ * s_, h_))
                    reshaped.append(x)
            deepstack_visual_embeds = reshaped

        cos_matrix, sin_matrix = self._prepare_cos_sin(rot_mats=rot_mats)
        assert cos_matrix.shape[2] >= start_pos + S
        cos_slice = cos_matrix[:, :, start_pos : start_pos + S, :]
        sin_slice = sin_matrix[:, :, start_pos : start_pos + S, :]
        if batch_size > 1:
            cos_slice = ttnn.reshape(cos_slice, [1, 1, B * S, cos_slice.shape[-1]])
            sin_slice = ttnn.reshape(sin_slice, [1, 1, B * S, sin_slice.shape[-1]])
        tt_rot_mats_prefill = [cos_slice, sin_slice]

        tt_page_table = (
            ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if page_table is not None
            else None
        )
        tt_chunk_page_table = (
            ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if chunk_page_table is not None
            else None
        )
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
        batch_size=1,
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
            batch_size=batch_size,
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
        batch_size=1,
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
                batch_size=batch_size,
            )
            if deepstack_visual_embeds is not None and i in range(len(deepstack_visual_embeds)):
                x = self.deepstack_process(x, deepstack_visual_embeds[i])

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        lm_head_norm_config = self.args.get_norm_config("lm_head", mode, None)
        x = self.norm(x, mode=mode, norm_config=lm_head_norm_config)

        if mode == Mode.PREFILL and self.args.get_lm_head_input_mem_config(mode, None).is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.args.get_lm_head_input_mem_config(mode, None))

        x = self.lm_head(x)

        if mode == Mode.PREFILL:
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x
