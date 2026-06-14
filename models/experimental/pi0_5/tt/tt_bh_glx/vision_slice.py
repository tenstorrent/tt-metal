# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chip SigLIP slices for the BH-Galaxy host-bounce pipeline.

Three slice classes wrap the existing TTNN SigLIP building blocks (no code
duplication — they import PatchEmbeddingTTNN / SigLIPBlockTTNN /
MultiModalProjectorTTNN). The block-sharded fast path (`forward_bs`) is
intentionally NOT used: BS requires entering once and exiting once around
the layer loop, which doesn't survive a host bounce. v1 uses the interleaved
`forward()` path for correctness; perf comes later.

Chip ownership (vision_per_chip[i]):
    chip 0: SigLIPEmbedSlice  (patch_embed + pos_emb, no transformer layers)
    chip 1: SigLIPLayerSlice  (layers 0..8)
    chip 2: SigLIPLayerSlice  (layers 9..17)
    chip 3: SigLIPTailSlice   (layers 18..26 + post_layernorm + mm_projector)
"""

from __future__ import annotations

from typing import Dict, Tuple

import os
import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn, get_ln_weight_memory_config
from models.experimental.pi0_5.tt.ttnn_siglip import (
    MultiModalProjectorTTNN,
    PatchEmbeddingTTNN,
    SigLIPBlockTTNN,
    _make_bs_memcfg,
    _SIGLIP_BS_GRID,
    _SIGLIP_INTERMEDIATE_PADDED,
)


def _glx_siglip_bs_enabled() -> bool:
    """Opt-in block-sharded fast path for the GLX vision slices.

    Each slice owns a contiguous run of layers on ONE chip, so BS can be
    entered once at slice start and exited once before the host-bounce send.
    The 'doesn't survive host bounce' concern in v1 only applies BETWEEN
    slices; WITHIN a slice the 9-layer loop stays on-chip. Default OFF.
    """
    return os.environ.get("PI0_GLX_SIGLIP_BS", "").lower() in ("1", "true", "yes", "on")


def _run_blocks_bs(blocks, hidden, config):
    """Enter BS once, run all blocks via forward_bs, exit BS to L1 interleaved.

    hidden: (B, num_patches, hidden) interleaved (as produced by embed slice
    or a host-bounce recv). Returns the same 3D interleaved shape so the
    downstream transport.send / post_ln see the v1 layout.
    """
    gx, gy = _SIGLIP_BS_GRID
    b, num_patches, hidden_dim = (int(d) for d in hidden.shape)
    total_m = b * num_patches
    mc_hidden = _make_bs_memcfg(1, total_m, hidden_dim, gx, gy)
    mc_qkv = _make_bs_memcfg(1, total_m, 144 * 32, gx, gy)
    mc_attn = _make_bs_memcfg(1, total_m, 48 * 32, gx, gy)
    mc_inter = _make_bs_memcfg(1, total_m, _SIGLIP_INTERMEDIATE_PADDED, gx, gy)

    x = ttnn.reshape(hidden, (1, 1, total_m, hidden_dim))
    x = ttnn.to_memory_config(x, mc_hidden, dtype=ttnn.bfloat16)
    for block in blocks:
        x = block.forward_bs(x, mc_hidden, mc_qkv, mc_attn, mc_inter, n_batch=b, n_seq=num_patches)
    x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.reshape(x, (b, num_patches, hidden_dim))


def _layer_weights(weights: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    """Extract per-layer weights — matches SigLIPVisionTowerTTNN._get_layer_weights."""
    prefixes = [f"vision_model.encoder.layers.{layer_idx}.", f"encoder.layers.{layer_idx}."]
    layer_weights = {}
    for prefix in prefixes:
        for key, value in weights.items():
            if key.startswith(prefix):
                layer_weights[key[len(prefix) :]] = value
    return layer_weights


class SigLIPEmbedSlice:
    """Chip 0: patch embedding + position embedding. No transformer layers."""

    def __init__(self, config: SigLIPConfig, vision_weights: Dict[str, torch.Tensor], submesh):
        self.config = config
        self.submesh = submesh
        self.patch_embed = PatchEmbeddingTTNN(config, vision_weights, submesh)

        pos_emb = vision_weights.get("position_embedding.weight") or vision_weights.get(
            "vision_model.embeddings.position_embedding.weight"
        )
        if pos_emb is None:
            raise RuntimeError("position_embedding.weight not found in vision weights")
        num_patches = (config.image_size // config.patch_size) ** 2
        if pos_emb.shape[0] != num_patches:
            raise RuntimeError(
                f"position embedding has {pos_emb.shape[0]} rows, expected {num_patches} for "
                f"image_size={config.image_size}, patch_size={config.patch_size}"
            )
        self.position_ids = ttnn.reshape(
            ttnn.arange(0, num_patches, 1, dtype=ttnn.uint32, device=submesh),
            (1, -1),
        )
        self.pos_emb_weights = ttnn.as_tensor(
            pos_emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, pixel_values) -> "ttnn.Tensor":
        """pixel_values: torch (B,3,H,W) or pre-folded ttnn tensor accepted by patch_embed.

        When PI0_SIGLIP_USE_FOLD=1 and PI0_SIGLIP_FOLD_HOST_PREP=1 and a torch
        (B,3,H,W) tensor is passed, we host-permute+reshape to the fold FAST
        PATH layout (B, H, W/patch, C*patch) ROW_MAJOR before upload. This
        moves the BCHW->BHWC permute + pixel untilize/reshape (tracy-measured
        ~1.8ms on the embed chip) off-device onto the host, where it is a
        cheap torch op overlapped with prior-stage compute.
        """
        if (
            getattr(self.patch_embed, "_use_fold", False)
            and os.environ.get("PI0_SIGLIP_FOLD_HOST_PREP", "").lower() in ("1", "true", "yes", "on")
            and isinstance(pixel_values, torch.Tensor)
            and pixel_values.dim() == 4
            and int(pixel_values.shape[1]) == self.patch_embed._fold_in_channels
        ):
            ps = self.config.patch_size
            B, C, H, W = (int(d) for d in pixel_values.shape)
            # (B,C,H,W) -> (B,H,W,C) -> (B, H, W/ps, C*ps) matching _forward_fold fast path (b)
            x = pixel_values.permute(0, 2, 3, 1).contiguous().reshape(B, H, W // ps, C * ps)
            pixel_values = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.submesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        hidden = self.patch_embed.forward(pixel_values)
        pos = ttnn.embedding(self.position_ids, self.pos_emb_weights, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(hidden, pos, memory_config=ttnn.L1_MEMORY_CONFIG)


class SigLIPLayerSlice:
    """Chips 1/2: a contiguous range of SigLIP transformer blocks."""

    def __init__(
        self,
        config: SigLIPConfig,
        vision_weights: Dict[str, torch.Tensor],
        submesh,
        layer_range: Tuple[int, int],
    ):
        self.config = config
        self.submesh = submesh
        self.layer_range = layer_range
        lo, hi = layer_range
        self.blocks = [SigLIPBlockTTNN(config, _layer_weights(vision_weights, i), submesh) for i in range(lo, hi)]

    def forward(self, hidden: "ttnn.Tensor") -> "ttnn.Tensor":
        if _glx_siglip_bs_enabled():
            return _run_blocks_bs(self.blocks, hidden, self.config)
        for block in self.blocks:
            hidden = block.forward(hidden)
        return hidden


class SigLIPTailSlice:
    """Chip 3: last layer range + post_layernorm + mm_projector → (B, 256, 2048)."""

    def __init__(
        self,
        config: SigLIPConfig,
        vision_weights: Dict[str, torch.Tensor],
        projector_weights: Dict[str, torch.Tensor],
        submesh,
        layer_range: Tuple[int, int],
    ):
        self.config = config
        self.submesh = submesh
        lo, hi = layer_range
        self.blocks = [SigLIPBlockTTNN(config, _layer_weights(vision_weights, i), submesh) for i in range(lo, hi)]

        post_ln_w = vision_weights.get("post_layernorm.weight") or vision_weights.get(
            "vision_model.post_layernorm.weight"
        )
        post_ln_b = vision_weights.get("post_layernorm.bias") or vision_weights.get("vision_model.post_layernorm.bias")
        if post_ln_w is None:
            raise RuntimeError("post_layernorm.weight not found in vision weights")
        mc = get_ln_weight_memory_config()
        self.post_ln_weight = tensor_1d_to_2d_ttnn(post_ln_w, submesh, dtype=ttnn.bfloat16, memory_config=mc)
        self.post_ln_bias = (
            tensor_1d_to_2d_ttnn(post_ln_b, submesh, dtype=ttnn.bfloat16, memory_config=mc)
            if post_ln_b is not None
            else None
        )

        self.mm_projector = MultiModalProjectorTTNN(projector_weights, submesh)

    def forward(self, hidden: "ttnn.Tensor") -> "ttnn.Tensor":
        if _glx_siglip_bs_enabled():
            hidden = _run_blocks_bs(self.blocks, hidden, self.config)
        else:
            for block in self.blocks:
                hidden = block.forward(hidden)
        hidden = ttnn.layer_norm(
            hidden,
            weight=self.post_ln_weight,
            bias=self.post_ln_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return self.mm_projector.forward(hidden)
