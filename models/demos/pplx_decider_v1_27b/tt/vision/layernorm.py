# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm(1152, eps 1e-6, affine + bias): the ViT ``norm1`` / ``norm2`` and the merger ``norm``."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.common import resolve_layer_norm
from models.demos.pplx_decider_v1_27b.tt.vision.weights import LayerNormWeights


@dataclass
class VisionLayerNormConfig:
    weights: LayerNormWeights
    eps: float
    optimizations: VisionOptimizations
    mesh_device: object | None = None


class VisionLayerNorm(LightweightModule):
    def __init__(self, weights: LayerNormWeights, eps: float, optimizations: VisionOptimizations):
        super().__init__()
        self.config = _resolve(VisionLayerNormConfig(weights=weights, eps=eps, optimizations=optimizations))
        self.weight = self.bias = None

    @classmethod
    def from_config(cls, config: VisionLayerNormConfig) -> "VisionLayerNorm":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance.weight = instance.bias = None
        return instance

    def load_device_weights(self) -> None:
        if self.weight is None:
            self.weight = self.config.weights.weight.get_device_weight()
            self.bias = self.config.weights.bias.get_device_weight()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        opts = self.config.optimizations
        return ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.config.eps,
            compute_kernel_config=opts.norm_compute_kernel_cfg,
            memory_config=opts.output_memcfg,
        )


def _resolve(config: VisionLayerNormConfig) -> VisionLayerNormConfig:
    device = config.mesh_device or config.optimizations.mesh_device
    return replace(config, mesh_device=device, weights=resolve_layer_norm(config.weights, device))
