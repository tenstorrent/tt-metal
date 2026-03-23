# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.tt.patch_merger import PatchMerger
from models.demos.qwen25_vl.tt.rope import RotarySetup
from models.demos.qwen25_vl.tt.vision_block import VisionBlock
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
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

    def prepare_input(self, patch_input, window_index, seq_len=None, window_pad_info=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            window_index (torch.Tensor): Window index tensor
            seq_len (int, optional): Padded sequence length
            window_pad_info (dict, optional): If non-uniform windows need padding to uniform size.
                Keys: orig_cu_seqlens (list), max_window_size (int), num_windows (int)

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        spatial_merge_unit = self.args.hf_config.vision_config.spatial_merge_size**2
        x = patch_input.reshape(patch_seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(patch_seq_len, -1)
        if window_pad_info is not None:
            orig_cu = window_pad_info["orig_cu_seqlens"]
            max_W = window_pad_info["max_window_size"]
            num_windows = window_pad_info["num_windows"]
            hidden_dim = x.shape[-1]
            new_x = torch.zeros(num_windows * max_W, hidden_dim, dtype=x.dtype)
            for i in range(num_windows):
                src_start = orig_cu[i]
                src_end = orig_cu[i + 1]
                size = src_end - src_start
                dst_start = i * max_W
                new_x[dst_start : dst_start + size] = x[src_start:src_end]
            x = new_x
            patch_seq_len = num_windows * max_W

        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
        x = self.args.prepare_residual_tensor_prefill(
            x,
            force_replicated=True,
        )
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
        cu_seqlens,
        cu_window_seqlens,
        windowed_window_info=None,
        full_window_info=None,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths
            cu_window_seqlens (torch.Tensor): Cumulative window sequence lengths
            rot_mats (list): Rotation matrices for positional embeddings
            windowed_window_info (dict): Window metadata for windowed attention layers (batched SDPA optimization)
            full_window_info (dict): Window metadata for full attention layers

        Returns:
            ttnn.Tensor: Output tensor
        """
        # Forward through each block
        for i, block in enumerate(self.blocks):
            # Determine which attention type to use (full or windowed)
            if i in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                window_info = full_window_info
            else:
                cu_seqlens_now = cu_window_seqlens
                window_info = windowed_window_info

            # Forward through block
            x = block(
                x,
                cu_seqlens=cu_seqlens_now,
                rot_mats=rot_mats,
                window_info=window_info,
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
    ):
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args

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
        """
        # process pixel_values for each image/video separately
        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []
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
            cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw,
                head_dim=self.model_args.vision_head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
                window_size=self.model_args.hf_config.vision_config.window_size,
                patch_size=self.model_args.hf_config.vision_config.patch_size,
            )

            # 2b. Compute window metadata for batched SDPA optimization.
            #     Instead of using windowed_scaled_dot_product_attention (O(S^2)),
            #     we batch windows into the batch dimension and use regular SDPA (O(S*W)).
            #     When windows are non-uniform, pad them to uniform max_W and apply a mask
            #     so real tokens don't attend to zero-padded positions.
            cu_window_seqlens_list = cu_window_seqlens.flatten().tolist()
            window_sizes = [
                cu_window_seqlens_list[i + 1] - cu_window_seqlens_list[i]
                for i in range(len(cu_window_seqlens_list) - 1)
            ]
            num_windows = len(window_sizes)
            windows_uniform = num_windows > 0 and all(w == window_sizes[0] for w in window_sizes)
            window_pad_info = None

            if windows_uniform:
                windowed_window_info = {
                    "uniform": True,
                    "window_size": window_sizes[0],
                    "num_windows": num_windows,
                }
            elif num_windows > 0:
                max_W = max(window_sizes)
                # Round up to multiple of 32 for tile alignment
                max_W = ((max_W + 31) // 32) * 32
                assert 2048 % max_W == 0, f"Aligned max window size {max_W} must divide 2048"

                window_pad_info = {
                    "orig_cu_seqlens": cu_window_seqlens_list,
                    "orig_window_sizes": window_sizes,
                    "max_window_size": max_W,
                    "num_windows": num_windows,
                }
                # Update unpadded_seq_len and seq_len for the padded layout
                unpadded_seq_len = num_windows * max_W
                seq_len = ((unpadded_seq_len // 2048) + 1) * 2048
                # Update cu_window_seqlens to uniform
                cu_window_seqlens = torch.tensor(
                    [i * max_W for i in range(num_windows + 1)], dtype=cu_window_seqlens.dtype
                )
                cu_seqlens = torch.tensor([0, unpadded_seq_len], dtype=cu_seqlens.dtype)

                total_windowed_windows = seq_len // max_W
                windowed_mask = torch.zeros(1, total_windowed_windows, max_W, max_W, dtype=torch.float32)
                for i in range(num_windows):
                    if window_sizes[i] < max_W:
                        windowed_mask[:, i, :, window_sizes[i] :] = float("-inf")
                tt_windowed_mask = ttnn.from_torch(
                    windowed_mask,
                    device=self.model_args.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat4_b,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
                )
                windowed_window_info = {
                    "uniform": True,
                    "window_size": max_W,
                    "num_windows": num_windows,
                    "attn_mask": tt_windowed_mask,
                }
                logger.info(
                    f"Padded {num_windows} non-uniform windows to uniform size {max_W} "
                    f"(original sizes: {sorted(set(window_sizes))})"
                )
            else:
                windowed_window_info = {"uniform": False}

            # Full-attention layers (4 of 32): use windowed SDPA with a single segment
            # covering the full (padded) sequence. No O(S²) mask needed — windowed SDPA
            # handles the cu_seqlens boundary natively.
            full_window_info = {"uniform": False}

            # 3. Use reference model's patch embedding
            patch_input = self.reference_model.patch_embed(pixel_values)

            # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)

            # If windows were padded to uniform size, remap position embeddings
            # so each window's real tokens get their original rotary embeddings
            # and padding positions get identity rotation (cos=1, sin=0).
            if window_pad_info is not None:
                orig_cu = window_pad_info["orig_cu_seqlens"]
                max_W = window_pad_info["max_window_size"]
                head_dim = cos_orig.shape[-1]
                new_cos = torch.ones(unpadded_seq_len, head_dim, dtype=cos_orig.dtype)
                new_sin = torch.zeros(unpadded_seq_len, head_dim, dtype=sin_orig.dtype)
                for i in range(num_windows):
                    src_start = orig_cu[i]
                    src_end = orig_cu[i + 1]
                    size = src_end - src_start
                    dst_start = i * max_W
                    new_cos[dst_start : dst_start + size] = cos_orig[src_start:src_end]
                    new_sin[dst_start : dst_start + size] = sin_orig[src_start:src_end]
                cos_orig = new_cos
                sin_orig = new_sin

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
            vision_mesh_mapper = ttnn.ReplicateTensorToMesh(self.model_args.mesh_device)
            cos = ttnn.from_torch(
                cos_padded,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=vision_mesh_mapper,
            )
            sin = ttnn.from_torch(
                sin_padded,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=vision_mesh_mapper,
            )
            rot_mats = [cos, sin]

            # 5. Prepare input tensor for the TT model using window_index
            tt_input = self.tt_model.prepare_input(patch_input, window_index, seq_len, window_pad_info)

            # --- TT Model Execution ---
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
                cu_seqlens=ttnn.from_torch(
                    cu_seqlens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.model_args.mesh_device
                ),
                cu_window_seqlens=ttnn.from_torch(
                    cu_window_seqlens,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.model_args.mesh_device,
                ),
                windowed_window_info=windowed_window_info,
                full_window_info=full_window_info,
            )

            # deallocate device tensors that are not needed by decode
            ttnn.deallocate(tt_input)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])
            if "attn_mask" in windowed_window_info:
                ttnn.deallocate(windowed_window_info["attn_mask"])

            # --- Postprocessing ---
            # 1. Convert TT output back to torch tensor
            tt_output_torch = ttnn.to_torch(
                tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=1)
            )

            # deallocate TT output
            ttnn.deallocate(tt_out)

            # 2. Extract the relevant output part and adjust shape (matching test logic)
            out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
            # Output shape from TT is [1, B=1, S, H_out_padded], slice H and squeeze B, batch dims
            tt_output_torch = tt_output_torch[:, 0:1, :, :out_hidden_size].squeeze(0).squeeze(0)

            # 2b. If windows were padded to uniform size, extract original token positions.
            #     After patch_merger, each window's tokens were reduced by spatial_merge_unit.
            #     Remove the inter-window padding tokens (which produced garbage output).
            if window_pad_info is not None:
                spatial_merge_unit = self.model_args.hf_config.vision_config.spatial_merge_size**2
                orig_cu = window_pad_info["orig_cu_seqlens"]
                max_W = window_pad_info["max_window_size"]
                merged_max_W = max_W // spatial_merge_unit
                parts = []
                for i in range(window_pad_info["num_windows"]):
                    orig_size = orig_cu[i + 1] - orig_cu[i]
                    merged_orig = orig_size // spatial_merge_unit
                    start = i * merged_max_W
                    parts.append(tt_output_torch[start : start + merged_orig, :])
                tt_output_torch = torch.cat(parts, dim=0)

            # 3. Apply reverse window indexing to match reference model output order
            reverse_indices = torch.argsort(window_index)
            final_output = tt_output_torch[reverse_indices, :]

            final_outputs.append(final_output)

        # concatenate all the outputs
        return torch.cat(final_outputs, dim=0)


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
        attention_class=None,
        rope_setup_class=None,
        prefetcher=None,
    ):
        if args.is_galaxy:
            self._apply_tg_patches(args)
            # Qwen2.5-VL vocab_size=152064 exceeds Galaxy default padded_vocab_size=131072.
            # On-device sampling computes per-device offsets from padded_vocab_size; an
            # undersized value produces wrong global token IDs from row 1+ → garbage output.
            if args.padded_vocab_size is not None and args.vocab_size > args.padded_vocab_size:
                rows = args.cluster_shape[0]
                align = rows * args.tile_size
                args.padded_vocab_size = math.ceil(args.vocab_size / align) * align

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

        if args.is_galaxy:
            self._undo_tg_patches(args)
            self._rebuild_tg_attention_biases(args, state_dict, weight_cache_path)
            self._upgrade_tg_decode_precision(args)
            self._replace_lm_head(args, mesh_device, dtype, state_dict, weight_cache_path)

    @staticmethod
    def _apply_tg_patches(args):
        """Monkey-patch DistributedNorm, head rearrangement, and configs before model creation."""
        import models.tt_transformers.tt.attention as attn_module
        import models.tt_transformers.tt.decoder as decoder_module
        import models.tt_transformers.tt.distributed_norm as dn_module
        import models.tt_transformers.tt.model as model_module
        from models.demos.qwen25_vl.tt.distributed_norm import QwenDistributedNorm

        Transformer._orig_dn_class = dn_module.DistributedNorm
        dn_module.DistributedNorm = QwenDistributedNorm
        decoder_module.DistributedNorm = QwenDistributedNorm
        model_module.DistributedNorm = QwenDistributedNorm

        Transformer._orig_needs_head_rearrangement = attn_module.Attention._needs_head_rearrangement
        attn_module.Attention._needs_head_rearrangement = lambda self: (
            self.args.device_name in ("T3K", "TG")
            and hasattr(self.args, "base_model_name")
            and self.args.base_model_name in ("Qwen2.5-VL-7B", "olmOCR-2-7B")
        )

        Transformer._pad_tg_head_counts(args)
        Transformer._patch_tg_configs(args)

    @staticmethod
    def _pad_tg_head_counts(args):
        """Pad n_heads/n_kv_heads so QKV weight rows align to head boundaries.

        On TG (8×4), the QKV weight is sharded across 8 rows via
        ShardTensor2dMesh(dims=(3,2)).  The attention module groups heads by
        num_devices_per_group = n_kv_heads.  For Qwen-7B (28Q, 4KV, head_dim=128):

            qkv_per_group = (7Q + 1K + 1V) × 128 = 1152
            qkv_size      = 4 × 1152 = 4608
            per_row        = 4608 / 8 = 576  ← NOT a multiple of head_dim!

        nlp_create_qkv_heads_decode infers head_dim = 576/9 = 64 (wrong).

        Fix: set n_kv_heads = n_rows so every row is its own device-group.
        Replicate each KV head to fill extra rows, pad Q heads with zeros:

            n_kv_heads  4 → 8   (each KV head duplicated to 2 rows)
            n_heads    28 → 32  (4 zero-padded Q heads, one per odd row)
            qkv_size         → 128 × (32 + 16) = 6144
            per_row           = 6144 / 8 = 768 = 6 × 128  ✓
        """
        n_rows = args.cluster_shape[0]
        if args.n_kv_heads >= n_rows:
            return

        Transformer._orig_n_heads = args.n_heads
        Transformer._orig_n_kv_heads = args.n_kv_heads
        Transformer._orig_qkv_size = args.qkv_size

        orig_gqa = args.n_heads // args.n_kv_heads
        devs_per_kv = n_rows // args.n_kv_heads
        heads_per_dev = math.ceil(orig_gqa / devs_per_kv)

        args.n_kv_heads = n_rows
        args.n_heads = heads_per_dev * n_rows
        args.qkv_size = args.head_dim * (2 * args.n_kv_heads + args.n_heads)

    @staticmethod
    def _pad_tg_state_dict(state_dict, args):
        """No-op: _rearrange_qkv_2d / _rearrange_wo / _rearrange_qkv_1d in the
        Attention module already handle Q zero-padding, K/V duplication, and Wo
        column padding via the head-order lists built by _head_rearrangement_params.
        Pre-padding the state_dict here would shift the indices those methods use,
        causing K/V heads to be mapped incorrectly (e.g. kv2/kv3 lost entirely).
        """

    @staticmethod
    def _undo_tg_patches(args):
        """Restore the original DistributedNorm and Attention classes."""
        import models.tt_transformers.tt.attention as attn_module
        import models.tt_transformers.tt.decoder as decoder_module
        import models.tt_transformers.tt.distributed_norm as dn_module
        import models.tt_transformers.tt.model as model_module

        orig = Transformer._orig_dn_class
        dn_module.DistributedNorm = orig
        decoder_module.DistributedNorm = orig
        model_module.DistributedNorm = orig

        attn_module.Attention._needs_head_rearrangement = Transformer._orig_needs_head_rearrangement

        if hasattr(Transformer, "_orig_n_heads"):
            args.n_heads = Transformer._orig_n_heads
            args.n_kv_heads = Transformer._orig_n_kv_heads
            args.qkv_size = Transformer._orig_qkv_size
            del Transformer._orig_n_heads
            del Transformer._orig_n_kv_heads
            del Transformer._orig_qkv_size

    def _rebuild_tg_attention_biases(self, args, state_dict, weight_cache_path):
        """Rebuild QKV bias tensors on each attention layer for TG (8,4) mesh.

        The base Attention.__init__ shards biases with ShardTensorToMesh across
        all 32 devices, which gives the wrong device-to-bias mapping on a 2D mesh.
        This method replaces them with correctly sharded biases:
          1. Chunked by num_devices_per_group (8 mesh rows), not 32
          2. Sharded with ShardTensor2dMesh(dims=(3, None)) -- across rows, replicated cols
          3. Scaled by 1/cs[1] (1/4) so the existing pre-all-reduce bias addition
             in the base forward methods yields the correct result after column-wise
             all-reduce: 4 * (bias/4) = bias
        """
        cs = args.cluster_shape
        bias_mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, None), mesh_shape=cs)
        needs_rearrangement = (
            args.device_name in ("T3K", "TG")
            and hasattr(args, "base_model_name")
            and args.base_model_name in ("Qwen2.5-VL-7B", "olmOCR-2-7B")
        )

        for layer_idx in range(args.n_layers):
            attn = self.layers[layer_idx].attention
            if attn.wqkv_bias_prefill is None:
                continue

            ttnn.deallocate(attn.wqkv_bias_prefill)
            for b in attn.wqkv_bias_decode:
                ttnn.deallocate(b)

            layer_name = args.get_state_dict_prefix("Attention", layer_idx)
            if args.dummy_weights or weight_cache_path is None:
                cache_name = lambda _: None
            else:
                cache_name = lambda name, _ln=layer_name: weight_cache_path / f"{_ln}.{name}"

            wq_bias = state_dict[f"{layer_name}.wq.bias"]
            wk_bias = state_dict[f"{layer_name}.wk.bias"]
            wv_bias = state_dict[f"{layer_name}.wv.bias"]

            if needs_rearrangement:
                wq_bias, wk_bias, wv_bias = attn._rearrange_qkv_1d(wq_bias, wk_bias, wv_bias)

            num_chunks = attn.num_devices_per_group
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(wq_bias, num_chunks)[i],
                            torch.chunk(wk_bias, num_chunks)[i],
                            torch.chunk(wv_bias, num_chunks)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(num_chunks)
                ],
                dim=-1,
            )
            qkv_bias = qkv_bias / cs[1]

            attn.wqkv_bias_prefill = ttnn.as_tensor(
                qkv_bias.reshape(1, 1, 1, -1),
                device=self.mesh_device,
                mesh_mapper=bias_mesh_mapper,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_prefill_tg_2d"),
            )

            attn.wqkv_bias_decode = []
            for batch_size in range(
                args.tile_size,
                args.max_batch_size + args.tile_size,
                args.tile_size,
            ):
                qkv_bias_decode = qkv_bias.unsqueeze(0).expand(batch_size, -1)
                bias_tensor = ttnn.as_tensor(
                    qkv_bias_decode.unsqueeze(0).unsqueeze(0),
                    device=self.mesh_device,
                    mesh_mapper=bias_mesh_mapper,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=cache_name(f"wqkv_bias_decode_tg_2d_{batch_size}"),
                )
                attn.wqkv_bias_decode.append(bias_tensor)

    def _upgrade_tg_decode_precision(self, args):
        """Upgrade attention precision from BFP8 to BF16 on TG.

        On TG, ccl_dtype defaults to BFP8, which is used for:
          - Prefill QKV matmul output and all-reduce
          - Prefill Wo all-reduce
          - Decode QKV matmul output and all-reduce

        activation_dtype defaults to None (→ BFP8) and controls:
          - Prefill Wo matmul output
          - Prefill SDPA Q typecast

        Upgrading both to BF16 improves the KV cache precision (K/V
        values written during prefill carry BF16 precision instead of
        BFP8), which reduces softmax-amplified errors in decode SDPA.
        The decode Wo path is also upgraded via explicit overrides.
        """
        for layer_idx in range(args.n_layers):
            attn = self.layers[layer_idx].attention
            attn.ccl_dtype = ttnn.bfloat16
            attn.activation_dtype = ttnn.bfloat16
            attn.wo_output_dtype = ttnn.bfloat16
            attn.wo_ccl_dtype = ttnn.bfloat16

    @staticmethod
    def _find_tile_compatible_grid(k_val, n_val, tile=32, max_dim=8):
        """Find a (rows, cols) grid where k_val%(tile*cols)==0 and n_val%(tile*rows)==0."""
        k_tiles = k_val // tile
        n_tiles = n_val // tile
        best_cols = 1
        for c in range(1, max_dim + 1):
            if k_tiles % c == 0:
                best_cols = c
        best_rows = 1
        for r in range(1, max_dim + 1):
            if n_tiles % r == 0:
                best_rows = r
        return (best_rows, best_cols)

    @staticmethod
    def _tiles_to_coregrid(num_tiles):
        """Convert tile count to a CoreGrid(y, x) with both dims <= 8."""
        n = num_tiles
        if n % 8 == 0:
            return ttnn.CoreGrid(y=n // 8, x=8)
        for y in range(1, 9):
            if n % y == 0 and n // y <= 8:
                return ttnn.CoreGrid(y=y, x=n // y)
        raise ValueError(f"Cannot create core grid for {n} cores within 8x8")

    @staticmethod
    def _patch_tg_configs(args):
        """Override MLP and attention configs that fail for dim=3584 on TG.

        The base TG configs in model_config.py are parameterised for models with
        tile-aligned dimension splits (e.g. Llama-70B dim=8192). Qwen-2.5-VL
        (dim=3584, hidden_dim=18944) produces non-tile-aligned shards when
        divided by the standard TG core counts.  This method overwrites every
        config that would produce a shard width not divisible by 32.
        """
        from functools import lru_cache

        from models.tt_transformers.tt.common import Mode

        dim = args.dim
        hidden_dim = args.hidden_dim
        cs = args.cluster_shape
        tile = 32
        qkv_size = args.qkv_size

        # ------------------------------------------------------------------ #
        # MLP program configs                                                 #
        # ------------------------------------------------------------------ #
        @lru_cache(maxsize=None)
        def get_mlp_ff1_3_prg_config(mode, seq_len=1, prefetcher=None):
            if mode == Mode.DECODE:
                return args.matmul_1d_config_from_tensor_shapes(
                    (1, 1, 32, dim // 4),
                    (1, 1, dim // 4, hidden_dim // 8),
                    grid=ttnn.CoreGrid(x=8, y=2),
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )
            elif mode == Mode.PREFILL:
                k_val = dim // cs[1]
                n_val = hidden_dim // cs[1]
                grid = Transformer._find_tile_compatible_grid(k_val, n_val)
                return args.matmul_config(
                    m=min(seq_len, args.prefill_len_cutoff),
                    k=k_val,
                    n=n_val,
                    grid_size=grid,
                )

        @lru_cache(maxsize=None)
        def get_mlp_ff2_prg_config(mode, seq_len=1, prefetcher=None):
            if mode == Mode.DECODE:
                return args.matmul_1d_config(
                    m=32,
                    k=hidden_dim // 8,
                    n=dim // 4,
                    grid=ttnn.CoreGrid(x=8, y=2),
                    overwrite_per_core_k=2,
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )
            elif mode == Mode.PREFILL:
                k_val = hidden_dim // cs[0]
                n_val = dim // cs[1]
                grid = Transformer._find_tile_compatible_grid(k_val, n_val)
                return args.matmul_config(
                    m=min(seq_len, args.prefill_len_cutoff),
                    k=k_val,
                    n=n_val,
                    grid_size=grid,
                )

        args.get_mlp_ff1_3_prg_config = get_mlp_ff1_3_prg_config
        args.get_mlp_ff2_prg_config = get_mlp_ff2_prg_config

        # ------------------------------------------------------------------ #
        # MLP all-reduce memory configs                                       #
        # ------------------------------------------------------------------ #
        # FF1/3 output per row device: hidden_dim/cs[0].
        # 18944/8 = 2368 = 74 tiles; 74 = 2*37, no grid ≤8×8 gives many cores.
        # Fall back to DRAM to avoid tile-alignment issues.
        args.model_config["FF1_OUT_GATHERED_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG

        # FF2 all-reduce output: dim/cs[1] = 896 = 28 tiles → 28 cores, shard=32
        dim_per_col = dim // cs[1]
        dim_per_col_tiles = dim_per_col // tile
        ff2_ar_grid = Transformer._tiles_to_coregrid(dim_per_col_tiles)
        ff2_ar_shard_cfg = lambda mesh_rows: ttnn.create_sharded_memory_config(
            shape=(tile * mesh_rows, tile),
            core_grid=ff2_ar_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        args.model_config["SELF_OUT_GATHERED_MEMCFG"] = ff2_ar_shard_cfg

        orig_ff2_ar = args.get_mlp_ff2_all_reduce_mem_config

        def get_mlp_ff2_all_reduce_mem_config(mode, tensor):
            if mode == Mode.DECODE and args.is_galaxy:
                return ff2_ar_shard_cfg(cs[0])
            return orig_ff2_ar(mode, tensor)

        args.get_mlp_ff2_all_reduce_mem_config = get_mlp_ff2_all_reduce_mem_config

        # ------------------------------------------------------------------ #
        # Attention QKV program config                                        #
        # ------------------------------------------------------------------ #
        orig_qkv_prg = args.get_attn_qkv_program_config

        def get_attn_qkv_program_config(mode, seq_len=1, prefetcher=None):
            if mode == Mode.DECODE and prefetcher is None:
                return None
            return orig_qkv_prg(mode, seq_len, prefetcher)

        args.get_attn_qkv_program_config = get_attn_qkv_program_config

        # ------------------------------------------------------------------ #
        # Attention QKV all-reduce output memory config                       #
        # ------------------------------------------------------------------ #
        qkv_per_row = qkv_size // cs[0]
        qkv_tiles = qkv_per_row // tile
        qkv_ar_grid = Transformer._tiles_to_coregrid(qkv_tiles)

        orig_qkv_ar = args.get_attn_qkv_all_reduce_output_mem_config

        def get_attn_qkv_all_reduce_output_mem_config(mode, mesh_cols=1, prefetcher=None):
            if mode == Mode.DECODE and prefetcher is None:
                return ttnn.create_sharded_memory_config(
                    (tile * mesh_cols, tile),
                    core_grid=qkv_ar_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            return orig_qkv_ar(mode, mesh_cols, prefetcher)

        args.get_attn_qkv_all_reduce_output_mem_config = get_attn_qkv_all_reduce_output_mem_config

        # ------------------------------------------------------------------ #
        # Attention gather-users memory config                                #
        # ------------------------------------------------------------------ #
        # After nlp_concat_heads_decode, width = n_local_heads * head_dim
        n_local_heads = args.n_heads // cs[0]
        gather_width = n_local_heads * args.head_dim
        gather_tiles = gather_width // tile
        gather_grid = Transformer._tiles_to_coregrid(gather_tiles)
        args.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
            shape=(tile * mesh_cols, tile),
            core_grid=gather_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # NOTE: Wo matmul uses the default core_grid=(4,8) auto-config.
        # The input is sharded on 8x2 (from user_selection_matrix), so
        # any program_config grid must contain that 8x2 grid.  Since
        # 28 N-tiles can't divide evenly across any grid with x>=8,y>=2,
        # we rely on ttnn's internal handling of non-integer tile distribution.

    def _replace_lm_head(self, args, mesh_device, dtype, state_dict, weight_cache_path):
        """Replace the base LMHead with a TG-compatible 2D-sharded version."""
        from models.demos.qwen25_vl.tt.lm_head import QwenLMHead

        self.lm_head = QwenLMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=ttnn.bfloat16,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", None),
            weight_cache_path=weight_cache_path,
        )

    def forward(
        self,
        x,
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
        batch_size=1,
    ):
        if mode == Mode.PREFILL and self.args.is_galaxy:
            ttnn.synchronize_device(self.mesh_device)

        return super().forward(
            x,
            current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            batch_size=batch_size,
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
            dtype=self.rope_setup.datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        host_sin = ttnn.from_torch(
            sin_matrix.expand(sin_matrix.shape[0], -1, -1, -1),
            device=None,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.rope_setup.datatype,
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

    def prepare_inputs_prefill(self, tokens, rot_mats, start_pos=0, page_table=None, chunk_page_table=None):
        assert isinstance(rot_mats[0], torch.Tensor)
        assert isinstance(rot_mats[1], torch.Tensor)
        assert tokens.dim() == 3, "tokens should be a 3D tensor"
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
