# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP slices for the 1×8 mesh pipeline.

Slice classes wrap the existing TTNN SigLIP building blocks (no code
duplication — they import PatchEmbeddingTTNN / SigLIPBlockTTNN /
MultiModalProjectorTTNN). `SigLIPCameraSlice` composes embed + all layers +
tail into the full encoder on one submesh and runs it camera-parallel: each
chip runs the whole encoder for its own camera(s) at small batch, in parallel
across the mesh (see SigLIPCameraSlice for the block-sharded encoder path).
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
    _SIGLIP_BS_GRID,
    _SIGLIP_INTERMEDIATE_PADDED,
    _make_bs_memcfg,
    _siglip_bs_enabled,
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
    so it needs no mesh mappers (each chip is a 1x1 submesh / single device).

    BS encoder path: since the whole encoder runs on one chip (no host bounce),
    we can use the same block-sharded forward_bs path that SigLIPVisionTowerTTNN
    uses. The bit-diff diagnostic showed the non-BS path diverges from the
    single-chip embed_image by max diff ~3.0 (PCC 0.9999), which compounds into
    the LIBERO N=5 accuracy gap. With BS, vision matches bit-for-bit.
    """

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
        self._bs_memcfgs_cache: Dict[Tuple[int, int], Tuple] = {}

    def _get_bs_memcfgs(self, b: int, m_padded: int):
        """Mirror SigLIPVisionTowerTTNN._get_bs_memcfgs: build the four
        block-sharded memcfgs the encoder data path uses."""
        key = (b, m_padded)
        if key not in self._bs_memcfgs_cache:
            gx, gy = _SIGLIP_BS_GRID
            total_m = b * m_padded
            mc_hidden = _make_bs_memcfg(1, total_m, self.config.hidden_size, gx, gy)
            mc_qkv = _make_bs_memcfg(1, total_m, 144 * 32, gx, gy)
            mc_attn = _make_bs_memcfg(1, total_m, 48 * 32, gx, gy)
            mc_intermediate = _make_bs_memcfg(1, total_m, _SIGLIP_INTERMEDIATE_PADDED, gx, gy)
            self._bs_memcfgs_cache[key] = (mc_hidden, mc_qkv, mc_attn, mc_intermediate)
        return self._bs_memcfgs_cache[key]

    def forward(self, pixel_values) -> "ttnn.Tensor":
        hidden = self.embed.forward(pixel_values)
        if _siglip_bs_enabled() and len(self.layers.blocks) > 0:
            # Enter BS once before the encoder loop, exit once after — mirrors
            # SigLIPVisionTowerTTNN.forward (ttnn_siglip.py:1529-1554).
            b, num_patches, hidden_dim = hidden.shape
            hidden = ttnn.reshape(hidden, (1, 1, int(b) * int(num_patches), int(hidden_dim)))
            mc_hidden, mc_qkv, mc_attn, mc_intermediate = self._get_bs_memcfgs(int(b), int(num_patches))
            hidden = ttnn.to_memory_config(hidden, mc_hidden, dtype=ttnn.bfloat16)
            for block in self.layers.blocks:
                hidden = block.forward_bs(
                    hidden,
                    mc_hidden,
                    mc_qkv,
                    mc_attn,
                    mc_intermediate,
                    n_batch=int(b),
                    n_seq=int(num_patches),
                )
            hidden = ttnn.sharded_to_interleaved(hidden, memory_config=ttnn.L1_MEMORY_CONFIG)
            hidden = ttnn.reshape(hidden, (int(b), int(num_patches), int(hidden_dim)))
        else:
            hidden = self.layers.forward(hidden)
        out = self.tail.forward(hidden)
        # Move final vision output to DRAM so L1 is freed before the
        # downstream prefill stage runs on the same chip — BS encoder leaves
        # more L1 residual than the non-BS path and the prefill MLP up_proj's
        # CBs need the headroom (else: "circular buffers clash with L1 buffers").
        if out.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            out_dram = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out)
            out = out_dram
        return out


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
