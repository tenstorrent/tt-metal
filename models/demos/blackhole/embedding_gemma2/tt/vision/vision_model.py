# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionModel: device forward of the vision path (HF `Gemma4VisionModel` + `embed_vision`).

    h      = patch_embedder(pixel_values, pos_x, pos_y)                 [B,1,S,768]
    mask   = repeat((keep - 1) * 1e9, S rows)                           [B,1,S,S]   key padding
    for i in 0..15:  h = layer_i(h, mask, cos, sin)
    pooled = (P @ h) * sqrt(768) / 9                                    [B,1,S_out,768] fp32
    soft   = RMSNorm_noscale(pooled) @ W_embed_vision                   [B,1,S_out,512] bf16

Contract (forward)
    `VisionDeviceInputs` built by `VisionEncoder.prepare_inputs` and uploaded (see vision_encoder.py):
        pixel_values [B,1,S,768] fp32 TILE, pos_x/pos_y [B,S] uint32 ROW_MAJOR, cos/sin [B,1,S,64]
        bf16 TILE, keep [B,1,1,S] bf16 TILE (1 valid patch / 0 padding), pool [B,1,S_out,S] bf16 TILE.
    S is a patch bucket (1280 or 2560), S_out = config.pooled_rows(S).
    Returns (soft_tokens [B,1,S_out,512] bf16, pooled [B,1,S_out,768] fp32). Image b's soft tokens
    are rows 0..n_out[b]-1 (n_out = valid patches / 9). forward() issues device ops only, so it can
    be captured in a trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import MASK_NEG
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionConfig
from models.demos.blackhole.embedding_gemma2.tt.vision.embedder import VisionEmbedder
from models.demos.blackhole.embedding_gemma2.tt.vision.encoder_layer import VisionEncoderLayer
from models.demos.blackhole.embedding_gemma2.tt.vision.patch_embedder import VisionPatchEmbedder
from models.demos.blackhole.embedding_gemma2.tt.vision.pooler import VisionPooler
from models.demos.blackhole.embedding_gemma2.tt.vision.rope import AxialRope
from models.demos.blackhole.embedding_gemma2.tt.weights import substate


@dataclass
class VisionDeviceInputs:
    pixel_values: ttnn.Tensor
    pos_x: ttnn.Tensor
    pos_y: ttnn.Tensor
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    keep: ttnn.Tensor
    pool: ttnn.Tensor

    def tensors(self) -> tuple[ttnn.Tensor, ...]:
        return (self.pixel_values, self.pos_x, self.pos_y, self.cos, self.sin, self.keep, self.pool)


def key_padding_mask(keep: ttnn.Tensor) -> ttnn.Tensor:
    """keep [B,1,1,S] (1 valid / 0 pad) -> additive [B,1,S,S]: 0 for valid keys, -1e9 for padding keys."""
    S = keep.shape[-1]
    dram = ttnn.DRAM_MEMORY_CONFIG
    shifted = ttnn.subtract(keep, 1.0, memory_config=dram)
    bias = ttnn.multiply(shifted, -MASK_NEG, memory_config=dram)
    ttnn.deallocate(shifted)
    mask = ttnn.repeat(bias, ttnn.Shape([1, 1, S, 1]), memory_config=dram)
    ttnn.deallocate(bias)
    return mask


class VisionModel(LightweightModule):
    def __init__(
        self,
        config: VisionConfig,
        device,
        patch_embedder: VisionPatchEmbedder,
        layers: list[VisionEncoderLayer],
        embedder: VisionEmbedder,
        rope: AxialRope,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.patch_embedder = patch_embedder
        self.layers = layers
        self.pooler = VisionPooler(config)
        self.embedder = embedder
        self.rope = rope

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: VisionConfig,
        device,
        *,
        cache_dir: Path | None = None,
        num_layers: int | None = None,
    ) -> "VisionModel":
        """`state_dict`: `vision_tower.*` and `embed_vision.*` keys (weights.load_vision_state_dict).

        num_layers < config.num_layers builds a truncated stack (debug / per-layer tests only).
        """
        n = config.num_layers if num_layers is None else num_layers
        tower = substate(state_dict, "vision_tower")
        rope = AxialRope.build(config.head_dim, device)
        patch = VisionPatchEmbedder.from_state_dict(
            substate(tower, "patch_embedder"), config, device, cache_dir=cache_dir
        )
        layers = [
            VisionEncoderLayer.from_state_dict(
                substate(tower, f"encoder.layers.{i}"), config.layer(i), device, rope, cache_dir=cache_dir
            )
            for i in range(n)
        ]
        embedder = VisionEmbedder.from_state_dict(
            substate(state_dict, "embed_vision"), config, device, cache_dir=cache_dir
        )
        return cls(config, device, patch, layers, embedder, rope)

    def forward_hidden(self, x: VisionDeviceInputs) -> ttnn.Tensor:
        """Patch embedding + 16 encoder layers. [B,1,S,768] bf16 (padding rows not meaningful)."""
        mask = key_padding_mask(x.keep)
        h = self.patch_embedder(x.pixel_values, x.pos_x, x.pos_y)
        for layer in self.layers:
            h_next = layer(h, mask, x.cos, x.sin)
            ttnn.deallocate(h)
            h = h_next
        ttnn.deallocate(mask)
        return h

    def forward(self, x: VisionDeviceInputs) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        h = self.forward_hidden(x)
        pooled = self.pooler(h, x.pool)
        ttnn.deallocate(h)
        soft = self.embedder(pooled)
        return soft, pooled
