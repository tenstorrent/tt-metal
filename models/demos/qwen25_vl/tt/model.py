# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.tt.patch_merger import PatchMerger
from models.demos.qwen25_vl.tt.rope import RotarySetup
from models.demos.qwen25_vl.tt.vision_block import VisionBlock
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model import Transformer as TTTransformer


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
        transformation_mat_torch = get_rot_transformation_mat(
            args.vision_head_dim
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
            dtype=dtype,
        )

    def prepare_input(self, patch_input, window_index, seq_len=None):
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
        cu_seqlens,
        cu_window_seqlens,
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

        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.vision_head_dim)
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

        The vision weights are replicated on every chip of the mesh, so the images are run data-parallel:
        each pass takes one image per chip (inputs, RoPE tables and window boundaries sharded on dim 0),
        padded to the longest image of the group. A group with fewer images than chips repeats its last
        image on the idle chips.
        """
        mesh = self.model_args.mesh_device
        # On a 32-chip Galaxy mesh the vision tower is tensor-parallel over the mesh columns (VisionModelArgs
        # splits the weights), so images go through one at a time with the hidden dim column-sharded; on
        # every other mesh the weights are replicated and the chips take one image each.
        image_parallel = not self.model_args.is_galaxy
        num_devices = mesh.get_num_devices() if image_parallel else 1
        # --- per-image host preprocessing ---
        images = []
        offset = 0
        for g in grid_thw:
            n_pix = int(g.prod())
            pv = pixel_values[offset : offset + n_pix]
            offset += n_pix
            g = g.unsqueeze(0)
            unpadded_seq_len = int((g[:, 1] * g[:, 2]).sum())
            # padded sequence length (multiple of 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048
            cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=g,
                head_dim=self.model_args.vision_head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
                window_size=self.model_args.hf_config.vision_config.window_size,
                patch_size=self.model_args.hf_config.vision_config.patch_size,
            )
            patch_input = self.reference_model.patch_embed(pv)
            cos, sin = convert_rope_style_hf_to_meta(*position_embeddings)
            # window-ordered patches, as the TT model consumes them
            spatial_merge_unit = self.model_args.hf_config.vision_config.spatial_merge_size**2
            x = patch_input.reshape(unpadded_seq_len // spatial_merge_unit, spatial_merge_unit, -1)[
                window_index
            ].reshape(unpadded_seq_len, -1)
            images.append(
                dict(
                    x=x,
                    cos=cos,
                    sin=sin,
                    cu_seqlens=cu_seqlens,
                    cu_window_seqlens=cu_window_seqlens,
                    window_index=window_index,
                    unpadded=unpadded_seq_len,
                    padded=seq_len,
                )
            )

        out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
        # Group images of similar length together so a pass is padded to the longest image of its group only
        order = sorted(range(len(images)), key=lambda i: images[i]["padded"])
        final_outputs = [None] * len(images)
        for group_start in range(0, len(images), num_devices):
            group_ids = order[group_start : group_start + num_devices]
            group = [images[i] for i in group_ids]
            n_real = len(group)
            group = group + [group[-1]] * (num_devices - n_real)  # idle chips repeat the last image
            S = max(im["padded"] for im in group)
            max_unpadded = max(im["unpadded"] for im in group)
            L = max(im["cu_seqlens"].numel() for im in group)
            Lw = max(im["cu_window_seqlens"].numel() for im in group)

            def _pad_rows(t, value=0.0):
                return torch.nn.functional.pad(t, (0, 0, 0, S - t.shape[-2]), value=value)

            def _pad_bounds(b, n):  # repeated trailing boundaries are empty windows (no-ops for the kernel)
                return torch.cat([b, b[-1:].expand(n - b.numel())]).to(torch.int32)

            x = torch.stack([_pad_rows(im["x"]) for im in group]).unsqueeze(1)  # [N, 1, S, H]
            cos = torch.stack([_pad_rows(im["cos"], 1.0) for im in group]).unsqueeze(1)  # [N, 1, S, D]
            sin = torch.stack([_pad_rows(im["sin"], 0.0) for im in group]).unsqueeze(1)
            cu = torch.stack([_pad_bounds(im["cu_seqlens"], L) for im in group])  # [N, L]
            cuw = torch.stack([_pad_bounds(im["cu_window_seqlens"], Lw) for im in group])  # [N, Lw]
            if image_parallel:
                shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)
                x_mapper = rope_mapper = bounds_mapper = shard0
            else:
                x_mapper = ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=self.model_args.cluster_shape)
                rope_mapper = bounds_mapper = ttnn.ReplicateTensorToMesh(mesh)
            common = dict(device=mesh, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=x_mapper, **common)
            rot_mats = [
                ttnn.from_torch(cos, dtype=ttnn.bfloat16, mesh_mapper=rope_mapper, **common),
                ttnn.from_torch(sin, dtype=ttnn.bfloat16, mesh_mapper=rope_mapper, **common),
            ]

            def _bounds_tensor(b):  # per-device 1-D uint32 row, as the windowed SDPA expects
                t = ttnn.from_torch(
                    b, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=bounds_mapper
                )
                return ttnn.reshape(t, (b.shape[-1],))

            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=max_unpadded,
                rot_mats=rot_mats,
                cu_seqlens=_bounds_tensor(cu),
                cu_window_seqlens=_bounds_tensor(cuw),
            )
            ttnn.deallocate(tt_input)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            # [N, 1, max_unpadded // merge, H_out_padded]
            if image_parallel:
                out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            else:
                out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))[:, 0:1]
            ttnn.deallocate(tt_out)
            for i, im in enumerate(group[:n_real]):
                merged_len = im["unpadded"] // spatial_merge_unit
                o = out[i, 0, :merged_len, :out_hidden_size]
                final_outputs[group_ids[i]] = o[torch.argsort(im["window_index"]), :]

        if self.debug:
            logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
            reference_output = self.reference_model.forward(pixel_values, grid_thw)
            reference_output = getattr(reference_output, "pooler_output", reference_output)
            _, pcc = comp_pcc(reference_output, torch.cat(final_outputs, dim=0))
            logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")

        return torch.cat(final_outputs, dim=0)


class Transformer(TTTransformer):
    # --- On-device greedy decode correctness on batch-32 (#48037) ---
    # Symptom: on-device sampling produced gibberish at batch-32 (BERTScore F1 ~0.34)
    # but is correct at batch-1, and host argmax of the same batch-32 run is correct
    # (F1 0.791) -> the decode forward is fine; only the on-device sampling path is wrong
    # at batch-32. Root cause: with allow_force_argmax disabled (the non-Galaxy default),
    # greedy decode (temperature=0 -> k=1,p=0,temp=1) goes through the heavy top-k/top-p
    # multi-all-gather sampling pipeline, which is what corrupts at batch-32, rather than
    # the simple single-gather argmax path.
    #
    # Fix: route greedy decode through the force-argmax path (enabled in
    # __init__ below) and run sampling eagerly so the all-gather re-acquires a
    # fresh semaphore. The decode token input cannot alias the sampling output.
    _tt_supports_decode_token_feedback = False
    _tt_disable_sampling_trace = True

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
        prefetcher=None,
    ):
        # Enable the single-gather force-argmax sampling path for greedy decode. The
        # non-Galaxy default (default_sampling_force_argmax) sets allow_force_argmax=False,
        # which forces greedy decode onto the heavy top-k/top-p pipeline that corrupts at
        # batch-32 (#48037). Must be set before super().__init__ builds the sampling module.
        ag_cfg = dict(args.model_config.get("SAMPLING_AG_CONFIG", {}) or {})
        ag_cfg["allow_force_argmax"] = True
        args.model_config["SAMPLING_AG_CONFIG"] = ag_cfg

        # Call parent constructor with vision-specific classes
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=Attention,
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
                    dtype=getattr(self.rope_setup, "datatype", ttnn.bfloat16),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                ),
            )
        return outputs

    def prepare_prefill_inputs_trace(self, tokens, rot_mats, page_table=None):
        """
        Prepare inputs for trace-based prefill. Returns host tensors (device=None)
        that can be copied to pre-allocated device tensors between trace replays.

        Args:
            tokens: [1, seq_len, hidden_dim] torch tensor (embeddings)
            rot_mats: (cos[1, 1, seq_len, head_dim], sin[1, 1, seq_len, head_dim]) torch tensors
            page_table: [1, num_blocks] torch tensor
        Returns:
            Tuple of host tensors: (tokens, cos, sin, page_table)
        """
        assert isinstance(rot_mats[0], torch.Tensor)
        assert isinstance(rot_mats[1], torch.Tensor)
        assert tokens.dim() == 3, "tokens should be a 3D tensor"

        host_tokens = ttnn.from_torch(
            tokens.unsqueeze(1),
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            ),
        )

        cos_matrix, sin_matrix = rot_mats[0], rot_mats[1]
        host_cos = ttnn.from_torch(
            cos_matrix.expand(cos_matrix.shape[0], -1, -1, -1),
            device=None,
            layout=ttnn.TILE_LAYOUT,
            dtype=getattr(self.rope_setup, "datatype", ttnn.bfloat16),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        host_sin = ttnn.from_torch(
            sin_matrix.expand(sin_matrix.shape[0], -1, -1, -1),
            device=None,
            layout=ttnn.TILE_LAYOUT,
            dtype=getattr(self.rope_setup, "datatype", ttnn.bfloat16),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        host_page_table = None
        if page_table is not None:
            host_page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (host_tokens, host_cos, host_sin, host_page_table)

    def prepare_inputs_prefill(
        self, tokens, rot_mats, start_pos=0, page_table=None, chunk_page_table=None, batch_size=1
    ):
        assert isinstance(rot_mats[0], torch.Tensor)
        assert isinstance(rot_mats[1], torch.Tensor)
        assert tokens.dim() == 3  # [B, seq_len, hidden_dim]
        B, S, H = tokens.shape
        if batch_size > 1:
            tokens = tokens.reshape(1, 1, B * S, H)
        else:
            tokens = tokens.unsqueeze(1)

        tokens_embd = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            ),
        )

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

        return tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    ):
        return super().ttnn_prefill_forward(
            x,
            rot_mats_global=rot_mats_global,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )
