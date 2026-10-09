# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decision readout: ``Linear(5120 -> 255, no bias)`` on the final-normed last real token.

The app takes ``last_hidden_state[:, -1]`` (after the final RMSNorm). The norm is row-wise, so
the caller may slice the last token before or after the norm. Masking by option count,
``/ temperature`` and softmax belong to the model head (stage 6), not to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.common import prefill_linear, resolve
from models.demos.pplx_decider_v1_27b.tt.optimizations import LinearOptimizations


@dataclass
class ReadoutConfig:
    weight: LazyWeight  # [5120, 255]
    linear: LinearOptimizations
    mesh_device: object | None = None


class PplxReadout(LightweightModule):
    def __init__(self, weight: LazyWeight, linear: LinearOptimizations, mesh_device=None):
        super().__init__()
        self.config = _resolve(ReadoutConfig(weight=weight, linear=linear, mesh_device=mesh_device))
        self.weight = None

    @classmethod
    def from_config(cls, config: ReadoutConfig) -> "PplxReadout":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance.weight = None
        return instance

    def load_device_weights(self) -> None:
        if self.weight is None:
            self.weight = self.config.weight.get_device_weight()

    def forward(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """hidden: [1, S, 5120] final-normed -> logits of the last token [1, 1, 255] BF16."""
        self.load_device_weights()
        length = hidden.shape[1]
        last = hidden[:, length - 1 : length, :] if length > 1 else hidden
        return prefill_linear(last, self.weight, "readout", self.config.linear)


def _resolve(config: ReadoutConfig) -> ReadoutConfig:
    device = config.mesh_device or config.weight.device or ttnn.GetDefaultDevice()
    return replace(config, mesh_device=device, weight=resolve(config.weight, device))
