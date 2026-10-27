# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm (HF `EmbeddingGemma2RMSNorm`).

    y = x * (mean(x^2, -1) + eps)^-0.5 * w        (with_scale=True;  w used as-is, NOT 1 + w)
    y = x * (mean(x^2, -1) + eps)^-0.5            (with_scale=False; v_norm)

Contract
    input   any-rank TILE tensor whose last dim == config.dim (bf16, DRAM interleaved)
    output  same shape / layout / dtype
    The reduction runs with HiFi4 + fp32 dest accumulation to match the fp32 HF norm.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig, compute_kernel_config_norm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, norm_weight


class RMSNorm(LightweightModule):
    def __init__(self, config: RMSNormConfig, weight: ttnn.Tensor | None):
        super().__init__()
        if config.with_scale and weight is None:
            raise ValueError("RMSNorm with_scale=True needs a weight")
        self.config = config
        self.weight = weight
        self.compute_kernel_config = compute_kernel_config_norm()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor] | None,
        config: RMSNormConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "norm",
    ) -> "RMSNorm":
        weight = None
        if config.with_scale:
            w = state_dict["weight"]
            if tuple(w.shape) != (config.dim,):
                raise ValueError(f"{name}: expected weight ({config.dim},), got {tuple(w.shape)}")
            weight = as_weight(
                norm_weight(w),
                device,
                cache_dir=cache_dir,
                name=f"{name}.weight",
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        return cls(config, weight)

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.config.eps,
            weight=self.weight,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
