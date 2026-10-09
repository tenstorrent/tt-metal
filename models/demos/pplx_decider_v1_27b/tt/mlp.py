# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP: ``down(silu(gate(x)) * up(x))``."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.common import prefill_linear, resolve
from models.demos.pplx_decider_v1_27b.tt.optimizations import LinearOptimizations
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import MLPWeights


@dataclass
class MLPConfig:
    weights: MLPWeights
    linear: LinearOptimizations
    mesh_device: object | None = None


class PplxMLP(LightweightModule):
    def __init__(self, weights: MLPWeights, linear: LinearOptimizations, mesh_device=None):
        super().__init__()
        self.config = _resolve(MLPConfig(weights=weights, linear=linear, mesh_device=mesh_device))
        self._loaded = False

    @classmethod
    def from_config(cls, config: MLPConfig) -> "PplxMLP":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance._loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        w = self.config.weights
        self.gate = w.gate.get_device_weight()
        self.up = w.up.get_device_weight()
        self.down = w.down.get_device_weight()
        self._loaded = True

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [B, S, 5120] BF16 TILE (already post-attention-normed)."""
        self.load_device_weights()
        opts = self.config.linear
        gate = prefill_linear(x, self.gate, "mlp_gate", opts)
        up = prefill_linear(x, self.up, "mlp_up", opts)
        product = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = prefill_linear(product, self.down, "mlp_down", opts)
        ttnn.deallocate(product)
        return out


def _resolve(config: MLPConfig) -> MLPConfig:
    device = config.mesh_device or config.weights.gate.device or ttnn.GetDefaultDevice()
    w = config.weights
    return replace(
        config,
        mesh_device=device,
        weights=MLPWeights(gate=resolve(w.gate, device), up=resolve(w.up, device), down=resolve(w.down, device)),
    )
