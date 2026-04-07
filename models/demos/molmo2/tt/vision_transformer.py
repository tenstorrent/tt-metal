# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Vision Transformer encoder for Molmo2.

This implements the full ViT encoder with:
- Patch embedding (14x14 patches from 378x378 images -> 729 tokens)
- Learned positional embedding (with bicubic interpolation for non-native sizes)
- 25 transformer blocks (27 total, 25 used)
- Multi-layer output collection for vision adapter

Key dimensions:
- Image size: 378x378
- Patch size: 14x14
- Number of patches: 729 (27x27)
- Hidden dim: 1152
- Heads: 16
- Head dim: 72

Multi-frame / multi-crop on a mesh: optional **frame-level data parallelism** via
``models.demos.molmo2.tt.frame_parallel_config`` (default milestone 1). Each parallel
round runs up to ``mesh_device.get_num_devices()`` frames (one per device); e.g. 32
frames on 8 devices => 4 equal rounds. If the frame count is not divisible by ``D``,
see ``MOLMO2_FRAME_DP_REMAINDER`` (``tail`` / ``pad`` / ``gather``).
"""

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.frame_parallel_config import (
    vision_frame_dp_enabled,
    vision_frame_dp_log_device_frames,
    vision_frame_dp_log_rounds,
    vision_frame_dp_remainder_mode,
)

_log = logging.getLogger(__name__)
from models.demos.molmo2.tt.vision_block import VisionBlock


def _mesh_physical_device_ids(mesh_device) -> List[int]:
    try:
        ids = list(mesh_device.get_device_ids())
        if ids:
            return ids
    except Exception:
        pass
    return list(range(mesh_device.get_num_devices()))


class VisionTransformer(LightweightModule):
    """
    Molmo2 Vision Transformer encoder.

    Processes images through patch embedding, positional embedding,
    and a stack of transformer blocks. Returns hidden states from
    all layers for multi-scale feature extraction.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        num_layers: int = 25,
        hidden_dim: int = 1152,
        intermediate_dim: int = 4304,
        num_heads: int = 16,
        head_dim: int = 72,
        patch_size: int = 14,
        image_size: int = 378,
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_vit",
        dtype=ttnn.bfloat8_b,
        allow_frame_parallel: bool = True,
    ):
        """
        Initialize VisionTransformer.

        Args:
            mesh_device: TTNN mesh device
            state_dict: Model state dict containing weights
            num_layers: Number of transformer blocks to use (25 for Molmo2)
            hidden_dim: Model hidden dimension (1152)
            intermediate_dim: MLP intermediate dimension
            num_heads: Number of attention heads (16)
            head_dim: Dimension per head (72)
            patch_size: Size of image patches (14)
            image_size: Expected input image size (378)
            layer_norm_eps: Epsilon for LayerNorm
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
            allow_frame_parallel: If False, skip frame-level data-parallel ViT on mesh (uses
                per-crop TTNN path). Required for mesh trace capture (frame DP uses host reads).
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.allow_frame_parallel = allow_frame_parallel
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.dtype = dtype

        # Calculate patch grid dimensions
        self.num_patches_per_side = image_size // patch_size  # 27
        self.num_patches = self.num_patches_per_side**2  # 729

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Patch embedding: Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # We store the weight and bias, actual embedding done on CPU or via TTNN fold+linear
        patch_embed_weight = state_dict[f"{state_dict_prefix}.patch_embedding.weight"]
        patch_embed_bias = state_dict[f"{state_dict_prefix}.patch_embedding.bias"]

        # Reshape conv weight for linear: [hidden_dim, 3*patch_size*patch_size]
        # Original shape: [hidden_dim, 3, patch_size, patch_size]
        self.patch_embed_weight_torch = patch_embed_weight.reshape(hidden_dim, -1).transpose(-2, -1)
        self.patch_embed_bias_torch = patch_embed_bias

        # Store on device for TTNN operations
        self.patch_embed_weight = ttnn.as_tensor(
            self.patch_embed_weight_torch.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("patch_embedding.weight"),
        )

        self.patch_embed_bias = ttnn.as_tensor(
            patch_embed_bias,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("patch_embedding.bias"),
        )

        # Positional embedding: [1, num_patches, hidden_dim]
        # Shape in state dict: [num_patches, hidden_dim]
        self.positional_embedding_torch = state_dict[f"{state_dict_prefix}.positional_embedding"]
        self.base_num_patches_per_side = self.num_patches_per_side  # For interpolation

        self.positional_embedding = ttnn.as_tensor(
            self.positional_embedding_torch.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("positional_embedding"),
        )

        # Transformer blocks
        self.blocks = []
        for layer_num in range(num_layers):
            block = VisionBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                layer_num=layer_num,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                layer_norm_eps=layer_norm_eps,
                weight_cache_path=weight_cache_path,
                state_dict_prefix=f"{state_dict_prefix}.transformer.resblocks.{layer_num}",
                dtype=dtype,
            )
            self.blocks.append(block)

    def interpolate_pos_embedding(
        self,
        target_patches_per_side: int,
    ) -> torch.Tensor:
        """
        Interpolate positional embedding to a different grid size.

        Uses bicubic interpolation (matching HuggingFace implementation).

        Args:
            target_patches_per_side: Target number of patches per side

        Returns:
            Interpolated positional embedding tensor
        """
        if target_patches_per_side == self.base_num_patches_per_side:
            return self.positional_embedding_torch

        # Reshape to 2D grid: [num_patches, hidden_dim] -> [1, h, w, hidden_dim]
        pos_embed = self.positional_embedding_torch.reshape(
            self.base_num_patches_per_side,
            self.base_num_patches_per_side,
            self.hidden_dim,
        )
        pos_embed = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)  # [1, hidden_dim, h, w]

        # Bicubic interpolation
        pos_embed = F.interpolate(
            pos_embed,
            size=(target_patches_per_side, target_patches_per_side),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape back: [1, hidden_dim, h, w] -> [num_patches, hidden_dim]
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim)

        return pos_embed

    def patch_embed_cpu(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Apply patch embedding on CPU.

        Args:
            pixel_values: Input images [batch, 3, height, width]

        Returns:
            Patch embeddings [batch, num_patches, hidden_dim]
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        # Unfold into patches: [batch, 3, patches_h, patch_size, patches_w, patch_size]
        x = pixel_values.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)

        # Reshape: [batch, patches_h * patches_w, patch_size * patch_size * 3]
        # After unfolds: [batch, C, patches_h, patches_w, patch_size, patch_size]
        # Need: [batch, patches_h, patches_w, patch_size, patch_size, C] (HWC order to match HF)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(
            batch_size, patches_h * patches_w, self.patch_size * self.patch_size * channels
        )

        # Linear projection (weight is already transposed to [588, 1152])
        x = torch.matmul(x, self.patch_embed_weight_torch) + self.patch_embed_bias_torch

        return x

    def patch_embed_ttnn(
        self,
        pixel_values: torch.Tensor,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Apply patch embedding with linear projection on TTNN.

        CPU handles only the unfold/permute (pure reshape, no compute).
        TTNN handles the linear projection and positional embedding add.

        Args:
            pixel_values: Input images [batch, 3, height, width]

        Returns:
            Embedded patches [1, 1, batch*num_patches, hidden_dim] on device
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        num_patches = patches_h * patches_w

        # CPU: unfold + permute -- pure data layout reorganization, no arithmetic
        x = pixel_values.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(
            1, 1, batch_size * num_patches, self.patch_size * self.patch_size * channels
        )

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        # Transfer raw (unprocessed) patches to device
        x_ttnn = ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # TTNN: linear projection -- x @ patch_embed_weight
        # x: [1, 1, B*N, 588],  patch_embed_weight: [1, 1, 588, 1152]
        embedded = ttnn.linear(
            x_ttnn,
            self.patch_embed_weight,
            bias=self.patch_embed_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_ttnn)

        # Add positional embedding: [1, 1, num_patches, 1152]
        if batch_size == 1:
            # Shapes match directly -- [1, 1, N, 1152] + [1, 1, N, 1152]
            embedded = ttnn.add(embedded, self.positional_embedding, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Tile positional embedding for multi-crop: [1, 1, B*N, 1152]
            pos_tiles = [self.positional_embedding] * batch_size
            pos_tiled = ttnn.concat(pos_tiles, dim=2)
            embedded = ttnn.add(embedded, pos_tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pos_tiled)

        return embedded

    def _is_mesh_device(self) -> bool:
        return self.mesh_device.__class__.__name__ == "MeshDevice"

    def _use_frame_level_dp(self, num_crops: int) -> bool:
        return self.allow_frame_parallel and vision_frame_dp_enabled() and self._is_mesh_device() and num_crops > 1

    def _replicated_embed_to_torch_crops(self, x: ttnn.Tensor, num_crops: int) -> torch.Tensor:
        seq_len = x.shape[2]
        hidden_dim = x.shape[3]
        if seq_len % num_crops != 0:
            raise ValueError(f"ViT seq len {seq_len} not divisible by num_crops={num_crops}")
        num_patches = seq_len // num_crops
        # Replicated mesh tensors cannot use bare to_torch (TT_FATAL: supply mesh composer).
        # All devices hold the same logical data — read any single-device shard.
        if self._is_mesh_device():
            th = ttnn.to_torch(ttnn.get_device_tensors(x)[0])
        else:
            th = ttnn.to_torch(x)
        if isinstance(th, (list, tuple)):
            th = th[0]
        return th.squeeze(0).squeeze(0).reshape(num_crops, num_patches, hidden_dim)

    def _gather_sharded_vit_layer_replicated(
        self,
        sharded_layer: ttnn.Tensor,
        chunk: int,
        num_patches: int,
        hidden_dim: int,
    ) -> ttnn.Tensor:
        th = ttnn.to_torch(
            sharded_layer,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1),
        )
        if isinstance(th, (list, tuple)):
            th = th[0]
        th = th.reshape(1, 1, chunk * num_patches, hidden_dim)
        return ttnn.from_torch(
            th,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _strip_frame_padding_from_hidden_list(
        self, layers: List[ttnn.Tensor], *, original_batch: int, num_patches: int
    ) -> List[ttnn.Tensor]:
        seq_keep = original_batch * num_patches
        out: List[ttnn.Tensor] = []
        for t in layers:
            if t.shape[2] <= seq_keep:
                out.append(t)
                continue
            hdim = t.shape[3]
            out.append(ttnn.slice(t, (0, 0, 0, 0), (1, 1, seq_keep, hdim)))
        return out

    def _log_frame_dp_device_map_sharded(
        self,
        mesh_device,
        cursor: int,
        chunk: int,
        *,
        round_kind: str,
        original_batch: int,
        remainder_mode: str,
    ) -> None:
        """Log shard slot -> physical device id -> global frame index (enable via env)."""
        if not vision_frame_dp_log_device_frames():
            return
        phy_ids = _mesh_physical_device_ids(mesh_device)
        parts: List[str] = []
        for slot in range(chunk):
            dev_id = phy_ids[slot] if slot < len(phy_ids) else slot
            gfi = cursor + slot
            pad_note = ""
            if remainder_mode == "pad" and gfi >= original_batch:
                pad_note = " [zero-pad]"
            parts.append("shard_%s->device_id=%s->global_frame=%s%s" % (slot, dev_id, gfi, pad_note))
        _log.info("ViT frame DP device map (%s): %s", round_kind, "; ".join(parts))

    def _log_frame_dp_tail_replicated(self, mesh_device, crop_idx: int) -> None:
        if not vision_frame_dp_log_device_frames():
            return
        phy_ids = _mesh_physical_device_ids(mesh_device)
        _log.info(
            "ViT frame DP tail (replicated): global_frame_index=%s on all %s devices; physical_device_ids=%s",
            crop_idx,
            len(phy_ids),
            phy_ids,
        )

    def _vit_frame_dp_run_sharded_slab(
        self,
        slab_1chn: torch.Tensor,
        chunk: int,
        num_patches: int,
        hidden_dim: int,
        *,
        return_all_hidden_states: bool,
        matmul_output_memory_config,
        shard_mapper,
        mesh_device,
        log_msg: str,
    ) -> List[ttnn.Tensor]:
        if vision_frame_dp_log_rounds():
            _log.info(log_msg)
        x_ttnn = ttnn.from_torch(
            slab_1chn,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_mapper,
        )
        hidden_sharded: List[ttnn.Tensor] = []
        for block in self.blocks:
            x_ttnn = block(x_ttnn, matmul_output_memory_config=matmul_output_memory_config)
            if return_all_hidden_states:
                hidden_sharded.append(x_ttnn)
        if not return_all_hidden_states:
            hidden_sharded = [x_ttnn]

        gathered: List[ttnn.Tensor] = []
        for h in hidden_sharded:
            g = self._gather_sharded_vit_layer_replicated(h, chunk, num_patches, hidden_dim)
            gathered.append(g)
            ttnn.deallocate(h)
        ttnn.deallocate(x_ttnn)
        return gathered

    def _vit_frame_parallel_from_torch(
        self,
        x_bt_nh: torch.Tensor,
        *,
        return_all_hidden_states: bool,
        matmul_output_memory_config,
    ) -> List[ttnn.Tensor]:
        """
        Frame-level data parallel ViT: each round uses ``ShardTensorToMesh(dim=1)`` with slab
        ``[1, chunk, N, H]``. Remainder handling is controlled by ``MOLMO2_FRAME_DP_REMAINDER``.
        """
        mesh_device = self.mesh_device
        num_dev = mesh_device.get_num_devices()
        original_batch, num_patches, hidden_dim = x_bt_nh.shape
        remainder_mode = vision_frame_dp_remainder_mode()

        pad_count = (num_dev - (original_batch % num_dev)) % num_dev
        if remainder_mode == "pad" and pad_count > 0:
            pad = torch.zeros(
                pad_count,
                num_patches,
                hidden_dim,
                dtype=x_bt_nh.dtype,
                device=x_bt_nh.device,
            )
            x_bt_nh = torch.cat([x_bt_nh, pad], dim=0)

        batch_size, num_patches, hidden_dim = x_bt_nh.shape
        shard_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

        round_outputs: List[List[ttnn.Tensor]] = []
        cursor = 0

        while cursor < batch_size:
            remaining = batch_size - cursor
            if remaining >= num_dev:
                chunk = num_dev
                slab = x_bt_nh[cursor : cursor + chunk].unsqueeze(0)
                self._log_frame_dp_device_map_sharded(
                    mesh_device,
                    cursor,
                    chunk,
                    round_kind="full_sharded_round",
                    original_batch=original_batch,
                    remainder_mode=remainder_mode,
                )
                gathered = self._vit_frame_dp_run_sharded_slab(
                    slab,
                    chunk,
                    num_patches,
                    hidden_dim,
                    return_all_hidden_states=return_all_hidden_states,
                    matmul_output_memory_config=matmul_output_memory_config,
                    shard_mapper=shard_mapper,
                    mesh_device=mesh_device,
                    log_msg=(
                        "ViT frame DP round: global_frames [%s, %s) chunk=%s devices=%s"
                        % (cursor, cursor + chunk, chunk, num_dev)
                    ),
                )
                round_outputs.append(gathered)
                cursor += chunk
            elif remainder_mode == "gather" and remaining > 0:
                chunk = remaining
                slab = x_bt_nh[cursor : cursor + chunk].unsqueeze(0)
                self._log_frame_dp_device_map_sharded(
                    mesh_device,
                    cursor,
                    chunk,
                    round_kind="partial_sharded_round (gather)",
                    original_batch=original_batch,
                    remainder_mode=remainder_mode,
                )
                gathered = self._vit_frame_dp_run_sharded_slab(
                    slab,
                    chunk,
                    num_patches,
                    hidden_dim,
                    return_all_hidden_states=return_all_hidden_states,
                    matmul_output_memory_config=matmul_output_memory_config,
                    shard_mapper=shard_mapper,
                    mesh_device=mesh_device,
                    log_msg=(
                        "ViT frame DP partial round (check-gather): global_frames [%s, %s) chunk=%s mesh_devices=%s"
                        % (cursor, cursor + chunk, chunk, num_dev)
                    ),
                )
                round_outputs.append(gathered)
                cursor += chunk
            else:
                all_crop_hidden: List[List[ttnn.Tensor]] = []
                for crop_idx in range(cursor, batch_size):
                    self._log_frame_dp_tail_replicated(mesh_device, crop_idx)
                    crop_x = x_bt_nh[crop_idx : crop_idx + 1].unsqueeze(0)
                    crop_ttnn = ttnn.from_torch(
                        crop_x,
                        device=mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=replicate_mapper,
                    )
                    states: List[ttnn.Tensor] = []
                    for block in self.blocks:
                        crop_ttnn = block(
                            crop_ttnn,
                            matmul_output_memory_config=matmul_output_memory_config,
                        )
                        if return_all_hidden_states:
                            states.append(crop_ttnn)
                    if not return_all_hidden_states:
                        states = [crop_ttnn]
                    all_crop_hidden.append(states)

                num_layers_out = len(all_crop_hidden[0])
                combined_tail: List[ttnn.Tensor] = []
                for layer_idx in range(num_layers_out):
                    parts = [ch[layer_idx] for ch in all_crop_hidden]
                    c = ttnn.concat(parts, dim=2)
                    combined_tail.append(c)
                    for t in parts:
                        ttnn.deallocate(t)
                if vision_frame_dp_log_rounds():
                    _log.info(
                        "ViT frame DP tail: sequential replicated frames [%s, %s)",
                        cursor,
                        batch_size,
                    )
                round_outputs.append(combined_tail)
                cursor = batch_size

        if len(round_outputs) == 1:
            out = round_outputs[0]
        else:
            num_layers_out = len(round_outputs[0])
            merged: List[ttnn.Tensor] = []
            for layer_idx in range(num_layers_out):
                parts = [rnd[layer_idx] for rnd in round_outputs]
                m = ttnn.concat(parts, dim=2)
                merged.append(m)
                for p in parts:
                    ttnn.deallocate(p)
            out = merged

        if remainder_mode == "pad" and pad_count > 0:
            out = self._strip_frame_padding_from_hidden_list(
                out, original_batch=original_batch, num_patches=num_patches
            )
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        patch_grid: Tuple[int, int] = None,
        return_all_hidden_states: bool = True,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_crops: int = 1,
    ) -> List[ttnn.Tensor]:
        """
        Forward through ViT blocks. With frame-level DP (see ``frame_parallel_config``),
        ``num_crops`` splits ``[1,1,B*N,H]`` into B rounds of parallel work on the mesh.
        """
        if self._use_frame_level_dp(num_crops):
            x_torch = self._replicated_embed_to_torch_crops(x, num_crops)
            return self._vit_frame_parallel_from_torch(
                x_torch,
                return_all_hidden_states=return_all_hidden_states,
                matmul_output_memory_config=matmul_output_memory_config,
            )

        hidden_states: List[ttnn.Tensor] = []
        for block in self.blocks:
            x = block(x, matmul_output_memory_config=matmul_output_memory_config)
            if return_all_hidden_states:
                hidden_states.append(x)

        if return_all_hidden_states:
            return hidden_states
        return [x]

    def forward_with_patch_embed(
        self,
        pixel_values: torch.Tensor,
        return_all_hidden_states: bool = True,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> List[ttnn.Tensor]:
        """
        Full forward pass including patch embedding (CPU) and transformer (TTNN).

        Args:
            pixel_values: Input images [batch, 3, height, width] as torch tensor
            return_all_hidden_states: If True, return hidden states from all layers

        Returns:
            List of hidden states from requested layers
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        x = self.patch_embed_cpu(pixel_values)

        if patches_h != self.base_num_patches_per_side or patches_w != self.base_num_patches_per_side:
            assert patches_h == patches_w, "Non-square patch grids not yet supported"
            pos_embed = self.interpolate_pos_embedding(patches_h)
        else:
            pos_embed = self.positional_embedding_torch

        x = x + pos_embed.unsqueeze(0)

        is_mesh = self._is_mesh_device()
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None

        if batch_size > 1 and self._use_frame_level_dp(batch_size):
            return self._vit_frame_parallel_from_torch(
                x,
                return_all_hidden_states=return_all_hidden_states,
                matmul_output_memory_config=matmul_output_memory_config,
            )

        if batch_size > 1:
            all_crop_hidden_states: List[List[ttnn.Tensor]] = []
            for crop_idx in range(batch_size):
                crop_x = x[crop_idx : crop_idx + 1].unsqueeze(0)
                crop_x_ttnn = ttnn.from_torch(
                    crop_x,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
                crop_hidden_states = self.forward(
                    crop_x_ttnn,
                    return_all_hidden_states=return_all_hidden_states,
                    matmul_output_memory_config=matmul_output_memory_config,
                )
                all_crop_hidden_states.append(crop_hidden_states)

            combined_hidden_states: List[ttnn.Tensor] = []
            num_layers_out = len(all_crop_hidden_states[0])
            for layer_idx in range(num_layers_out):
                layer_outputs = [s[layer_idx] for s in all_crop_hidden_states]
                combined = ttnn.concat(layer_outputs, dim=2)
                combined_hidden_states.append(combined)
                for t in layer_outputs:
                    ttnn.deallocate(t)
            return combined_hidden_states

        x = x.unsqueeze(0)
        x_ttnn = ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return self.forward(
            x_ttnn,
            return_all_hidden_states=return_all_hidden_states,
            matmul_output_memory_config=matmul_output_memory_config,
        )
