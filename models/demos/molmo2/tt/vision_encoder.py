# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 ViT encoder: 25 blocks, captures hidden states at layers 18 and 24.

Data-parallel across T3K: pixel_values are sharded along the batch (crops) dim
using ShardTensorToMesh, while ViT weights are replicated (ReplicateTensorToMesh).
After all 25 blocks, hidden states from layers 24 and 18 are concatenated along
the feature dim to produce [B*crops, 729, 2304] image features.

Absolute positional embedding (bicubic interpolated for non-27×27 grids) is
added AFTER the linear patch embedding.
"""

import math

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.vision_block import TtMolmo2ViTBlock


class TtMolmo2ViTEncoder(LightweightModule):
    """Molmo2 ViT encoder, data-parallel across T3K."""

    def __init__(self, mesh_device, state_dict, vit_cfg, weight_cache_path):
        super().__init__()

        self.mesh_device = mesh_device
        self.vit_cfg = vit_cfg
        self.capture_layers = vit_cfg.vit_capture_layers  # (24, 18)
        self.n_layers = vit_cfg.vit_n_layers  # 25
        self.vit_hidden = vit_cfg.vit_hidden  # 1152
        self.pos_embed_seq = vit_cfg.vit_pos_embed_seq  # 729

        vit_prefix = "model.vision_backbone.image_vit"

        if vit_cfg.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"molmo2_vit.{name}"

        # ------------------------------------------------------------------ #
        # Patch embedding: Linear(588, 1152) with bias
        # weight [1152, 588] → transpose to [1, 1, 588, 1152] for TTNN matmul
        # ------------------------------------------------------------------ #
        patch_emb_w = state_dict[f"{vit_prefix}.patch_embedding.weight"].T  # [588, 1152]
        self.patch_emb_weight = ttnn.as_tensor(
            patch_emb_w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("patch_emb_weight"),
        )
        self.patch_emb_bias = ttnn.as_tensor(
            state_dict[f"{vit_prefix}.patch_embedding.bias"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("patch_emb_bias"),
        )

        # ------------------------------------------------------------------ #
        # Positional embedding [729, 1152] — stored on CPU for optional bicubic
        # interpolation, then uploaded to device
        # ------------------------------------------------------------------ #
        self._pos_emb_cpu = state_dict[f"{vit_prefix}.positional_embedding"].float()
        self._current_patch_num = None
        self._pos_emb_ttnn = None
        self._upload_pos_emb((27, 27))

        # ------------------------------------------------------------------ #
        # ViT blocks
        # ------------------------------------------------------------------ #
        self.blocks = [
            TtMolmo2ViTBlock(mesh_device, state_dict, i, vit_cfg, weight_cache_path) for i in range(self.n_layers)
        ]

    def _upload_pos_emb(self, patch_num):
        """Upload positional embedding to device, bicubic-interpolating if needed."""
        if patch_num == self._current_patch_num and self._pos_emb_ttnn is not None:
            return

        grid_h = grid_w = int(math.sqrt(self._pos_emb_cpu.shape[0]))
        pe = self._pos_emb_cpu.reshape(grid_h, grid_w, self._pos_emb_cpu.shape[1])
        ph, pw = patch_num

        if ph != grid_h or pw != grid_w:
            pe = pe.unsqueeze(0).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(ph, pw), mode="bicubic", align_corners=False, antialias=True)
            pe = pe.permute(0, 2, 3, 1).squeeze(0)

        pe = pe.reshape(1, 1, ph * pw, self._pos_emb_cpu.shape[1]).to(torch.bfloat16)  # [1,1,729,1152]

        self._pos_emb_ttnn = ttnn.from_torch(
            pe,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._current_patch_num = patch_num

    def forward(self, pixel_values: ttnn.Tensor, patch_num=(27, 27)) -> ttnn.Tensor:
        """Run ViT encoder and return concatenated features from layers 18 and 24.

        Args:
            pixel_values: [B*n_crops, 729, 588] on device (sharded across crops)
            patch_num: (h, w) patch grid — (27, 27) for standard crops

        Returns:
            image_features: [B*n_crops, 729, 2304] on device (replicated after AllGather)
        """
        self._upload_pos_emb(patch_num)

        # pixel_values arrives as [n_crops, 1, 729, 588] (pre-shaped in run_vision_backbone).
        # TP ViT: all crops replicated on all devices; weights sharded for head-level TP.
        n_crops = pixel_values.shape[0]
        n_patches = pixel_values.shape[2]
        x = ttnn.to_layout(pixel_values, ttnn.TILE_LAYOUT)

        # Patch embedding: [n_crops, 1, 729, 1152]
        x = ttnn.linear(
            x,
            self.patch_emb_weight,
            bias=self.patch_emb_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.vit_cfg.compute_kernel_config_hifi2_fp16,
        )

        # Add positional embedding: tile for n_crops → [n_crops, 1, 729, 1152]
        if n_crops == 1:
            x = ttnn.add(x, self._pos_emb_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            pos_tiles = [self._pos_emb_ttnn] * n_crops
            pos_tiled = ttnn.concat(pos_tiles, dim=0)  # [n_crops, 1, 729, 1152]
            x = ttnn.add(x, pos_tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Run 25 ViT blocks. Input: [n_crops, 1, 729, 1152].
        # Each block does per-crop SDPA (fuse_batch=False keeps crops independent).
        # TP: each device computes 2 of 16 heads; ttnn.all_reduce combines after wo/MLP.
        captured = {}
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            if i in self.capture_layers:
                captured[i] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                # x is NOT deallocated by the block — we hold it for the next iteration

        # Concatenate captured features along last dim → [n_crops, 1, 729, 2304]
        feats = [ttnn.to_layout(captured[layer], ttnn.ROW_MAJOR_LAYOUT) for layer in self.capture_layers]
        image_features = ttnn.concat(feats, dim=-1)
        for f in feats:
            ttnn.deallocate(f)
        ttnn.deallocate(x)

        # Reshape to [1, n_crops, 729, 2304] for consistent downstream format
        image_features = ttnn.to_layout(image_features, ttnn.ROW_MAJOR_LAYOUT)
        image_features = ttnn.reshape(image_features, [1, n_crops, n_patches, 2304])
        image_features = ttnn.to_layout(image_features, ttnn.TILE_LAYOUT)
        return image_features  # [1, n_crops, 729, 2304] replicated on all T3K devices
