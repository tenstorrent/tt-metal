# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Patch embedding + learned position embedding (HF ``Qwen3_5VisionModel.forward`` 1103-1108).

HF: ``Conv3d(3, 1152, kernel = stride = (2, 16, 16), bias)`` over ``pixel_values.view(-1, 3, 2, 16, 16)``.
Kernel == stride, so it is exactly one matmul ``[S, 1536] x [1536, 1152] + bias`` on the processor's
flattened (C, T, H, W) patch rows. Then ``+ pos_embeds.to(bf16)``, the bilinear interpolation of the
48 x 48 learned table to the image grid, which depends only on ``grid_thw`` and is computed on host
when the image is prepared (``tower.prepare_inputs``), with HF's own helper and op order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.common import device_linear, resolve_linear, vision_linear
from models.demos.pplx_decider_v1_27b.tt.vision.weights import LinearWeights


@dataclass
class PatchEmbedConfig:
    weights: LinearWeights  # [1536, 1152] + [1, 1152]
    optimizations: VisionOptimizations
    mesh_device: object | None = None


class VisionPatchEmbed(LightweightModule):
    def __init__(self, weights: LinearWeights, optimizations: VisionOptimizations):
        super().__init__()
        self.config = _resolve(PatchEmbedConfig(weights=weights, optimizations=optimizations))
        self._loaded = False

    @classmethod
    def from_config(cls, config: PatchEmbedConfig) -> "VisionPatchEmbed":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance._loaded = False
        return instance

    def load_device_weights(self) -> None:
        if not self._loaded:
            self.weight, self.bias = device_linear(self.config.weights)
            self._loaded = True

    def embed(self, pixels: ttnn.Tensor) -> ttnn.Tensor:
        """Conv3d as a matmul: pixels [1, 1, S, 1536] BF16 -> [1, 1, S, 1152]."""
        self.load_device_weights()
        return vision_linear(pixels, self.weight, self.bias, "vision_patch", self.config.optimizations)

    def forward(self, pixels: ttnn.Tensor, pos_embed: ttnn.Tensor) -> ttnn.Tensor:
        """``patch_embed(pixels) + pos_embed``; pos_embed [1, 1, S, 1152] BF16 (zero on padded rows)."""
        h = self.embed(pixels)
        out = ttnn.add(h, pos_embed, memory_config=self.config.optimizations.output_memcfg)
        ttnn.deallocate(h)
        return out


def _resolve(config: PatchEmbedConfig) -> PatchEmbedConfig:
    device = config.mesh_device or config.optimizations.mesh_device
    return replace(config, mesh_device=device, weights=resolve_linear(config.weights, device))
