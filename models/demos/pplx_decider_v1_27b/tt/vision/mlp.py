# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ViT MLP: ``fc2(gelu_pytorch_tanh(fc1(x)))``, 1152 -> 4304 (padded 4320) -> 1152, both with bias.

HF ``Qwen3_5VisionMLP`` uses ``ACT2FN["gelu_pytorch_tanh"]`` (config ``hidden_act``), the tanh
approximation. The fused ttnn activation string for it is ``"gelu_tanh"`` (``UnaryOpType::GELU_TANH``,
``unary_op_utils.cpp`` ``string_to_unary_with_param``); ``"gelu"`` is the exact erf variant, used by
the merger instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.common import device_linear, resolve_linear, vision_linear
from models.demos.pplx_decider_v1_27b.tt.vision.weights import VisionMLPWeights

MLP_ACTIVATION = "gelu_tanh"  # gelu_pytorch_tanh


@dataclass
class VisionMLPConfig:
    weights: VisionMLPWeights
    optimizations: VisionOptimizations
    mesh_device: object | None = None


class VisionMLP(LightweightModule):
    def __init__(self, weights: VisionMLPWeights, optimizations: VisionOptimizations):
        super().__init__()
        self.config = _resolve(VisionMLPConfig(weights=weights, optimizations=optimizations))
        self._loaded = False

    @classmethod
    def from_config(cls, config: VisionMLPConfig) -> "VisionMLP":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance._loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        self.fc1_w, self.fc1_b = device_linear(self.config.weights.fc1)
        self.fc2_w, self.fc2_b = device_linear(self.config.weights.fc2)
        self._loaded = True

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, S, 1152] (norm2 output)."""
        self.load_device_weights()
        opts = self.config.optimizations
        h = vision_linear(x, self.fc1_w, self.fc1_b, "vision_fc1", opts, activation=MLP_ACTIVATION)
        out = vision_linear(h, self.fc2_w, self.fc2_b, "vision_fc2", opts)
        ttnn.deallocate(h)
        return out


def _resolve(config: VisionMLPConfig) -> VisionMLPConfig:
    device = config.mesh_device or config.optimizations.mesh_device
    w = config.weights
    return replace(
        config,
        mesh_device=device,
        weights=VisionMLPWeights(fc1=resolve_linear(w.fc1, device), fc2=resolve_linear(w.fc2, device)),
    )
