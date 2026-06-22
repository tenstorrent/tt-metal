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

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn, get_ln_weight_memory_config
from models.experimental.pi0_5.tt.ttnn_siglip import (
    MultiModalProjectorTTNN,
    PatchEmbeddingTTNN,
    SigLIPBlockTTNN,
)


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
        """pixel_values: torch (B,3,H,W) or pre-folded ttnn tensor accepted by patch_embed."""
        hidden = self.patch_embed.forward(pixel_values)
        pos = ttnn.embedding(self.position_ids, self.pos_emb_weights, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(hidden, pos, memory_config=ttnn.L1_MEMORY_CONFIG)


class SigLIPCameraSlice:
    """FULL SigLIP on ONE chip: embed + all layers + post_layernorm + mm_projector
    → (B, 256, 2048). For camera-parallel data parallelism — each chip runs the whole
    encoder for its own camera(s) at small batch, in parallel, with NO inter-layer
    socket hops (unlike the layer-sliced path). Composes the existing per-chip blocks,
    so it needs no mesh mappers (each chip is a 1x1 submesh / single device)."""

    def __init__(
        self,
        config: SigLIPConfig,
        vision_weights: Dict[str, torch.Tensor],
        projector_weights: Dict[str, torch.Tensor],
        submesh,
    ):
        self.config = config
        self.submesh = submesh
        self.embed = SigLIPEmbedSlice(config, vision_weights, submesh)
        self.layers = SigLIPLayerSlice(config, vision_weights, submesh, layer_range=(0, config.num_hidden_layers))
        # Empty layer range → SigLIPTailSlice contributes only post_ln + projector.
        self.tail = SigLIPTailSlice(
            config,
            vision_weights,
            projector_weights,
            submesh,
            layer_range=(config.num_hidden_layers, config.num_hidden_layers),
        )

    def forward(self, pixel_values) -> "ttnn.Tensor":
        hidden = self.embed.forward(pixel_values)
        hidden = self.layers.forward(hidden)
        return self.tail.forward(hidden)


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
