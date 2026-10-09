# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Patch merger (HF ``Qwen3_5VisionPatchMerger``, ``use_postshuffle_norm=False``): 2x2 patches -> one 5120 token.

``LayerNorm(1152)`` per patch, then ``view(-1, 4608)``: the processor emits patches in merge-block
order, so 4 consecutive rows are one 2x2 block. ``fc1 4608 -> 4608`` + ``nn.GELU()`` (exact erf, NOT
the tanh form the ViT MLP uses; ttnn activation ``"gelu"``), ``fc2 4608 -> 5120``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.common import device_linear, resolve_linear, vision_linear
from models.demos.pplx_decider_v1_27b.tt.vision.config import PplxVisionArgs
from models.demos.pplx_decider_v1_27b.tt.vision.layernorm import VisionLayerNorm, VisionLayerNormConfig
from models.demos.pplx_decider_v1_27b.tt.vision.weights import PatchMergerWeights

MERGER_ACTIVATION = "gelu"  # nn.GELU() default approximate="none"


@dataclass
class PatchMergerConfig:
    weights: PatchMergerWeights
    args: PplxVisionArgs
    optimizations: VisionOptimizations
    mesh_device: object | None = None


class VisionPatchMerger(LightweightModule):
    def __init__(self, config: PatchMergerConfig):
        super().__init__()
        device = config.mesh_device or config.optimizations.mesh_device
        w = config.weights
        self.config = replace(
            config,
            mesh_device=device,
            weights=PatchMergerWeights(
                norm=w.norm, fc1=resolve_linear(w.fc1, device), fc2=resolve_linear(w.fc2, device)
            ),
        )
        self.norm = VisionLayerNorm.from_config(
            VisionLayerNormConfig(w.norm, config.args.layer_norm_eps, config.optimizations, device)
        )
        self._loaded = False

    @classmethod
    def from_config(cls, config: PatchMergerConfig) -> "VisionPatchMerger":
        return cls(config)

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        self.norm.load_device_weights()
        self.fc1_w, self.fc1_b = device_linear(self.config.weights.fc1)
        self.fc2_w, self.fc2_b = device_linear(self.config.weights.fc2)
        self._loaded = True

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, S, 1152] (last block output), S % 128 == 0 -> [1, 1, S/4, 5120]."""
        self.load_device_weights()
        a, opts = self.config.args, self.config.optimizations
        s = x.shape[-2]
        normed = self.norm(x)
        grouped = ttnn.reshape(normed, (1, 1, s // a.merge_unit, a.merger_hidden_size))
        h = vision_linear(grouped, self.fc1_w, self.fc1_b, "merger_fc1", opts, activation=MERGER_ACTIVATION)
        ttnn.deallocate(grouped)
        out = vision_linear(h, self.fc2_w, self.fc2_b, "merger_fc2", opts)
        ttnn.deallocate(h)
        return out
