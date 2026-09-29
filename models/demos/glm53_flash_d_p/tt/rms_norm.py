# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 RMSNorm (input_layernorm and friends) on device, replicated: y = w * x * (mean(x^2) + eps)^-0.5.

From models/demos/mimo_v2_6_d_p/tt/rms_norm.py:TtRMSNorm: plain w (not 1 + w) as a TILE [1, 1, 1, H] gamma,
ttnn.bringup.rms_norm at HiFi4 + fp32 dest (sum of squares and normalization in fp32 inside the kernel).
GLM_NORM_IMPL=native selects ttnn.rms_norm for comparison.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate


class TtRMSNorm(LightweightModule):
    def __init__(self, mesh, weight: torch.Tensor, eps: float = 1e-5):
        self.eps = float(eps)
        self.weight = replicate(mesh, weight.float().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)
        self.ckc = hifi4_config()
        self.op = ttnn.rms_norm if os.environ.get("GLM_NORM_IMPL", "bringup") == "native" else ttnn.bringup.rms_norm

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x replicated [1, 1, S, H] TILE -> same shape, replicated, x's dtype."""
        return self.op(
            x,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def build_norm(mesh, loader, cfg, layer: int, name: str = "input_layernorm") -> TtRMSNorm:
    return TtRMSNorm(mesh, loader.layer(layer, f"{name}.weight"), cfg.rms_norm_eps)
