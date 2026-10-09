# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zero-centred RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``. The ``+1`` is folded at setup."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.common import resolve
from models.demos.pplx_decider_v1_27b.tt.optimizations import NormOptimizations


@dataclass
class RMSNormConfig:
    weight: LazyWeight  # (1 + w), [1, 1, D]
    eps: float
    mesh_device: object | None = None
    compute_kernel_cfg: object | None = None
    output_memcfg: ttnn.MemoryConfig | None = None


class PplxRMSNorm(LightweightModule):
    def __init__(self, weight: LazyWeight, eps: float, optimizations: NormOptimizations | None = None):
        super().__init__()
        config = RMSNormConfig(weight=weight, eps=eps)
        if optimizations is not None:
            config = replace(
                config,
                compute_kernel_cfg=optimizations.compute_kernel_cfg,
                output_memcfg=optimizations.output_memcfg,
            )
        self.config = _resolve(config)
        self.weight = None

    @classmethod
    def from_config(cls, config: RMSNormConfig) -> "PplxRMSNorm":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance.weight = None
        return instance

    def load_device_weights(self) -> None:
        if self.weight is None:
            self.weight = self.config.weight.get_device_weight()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.config.eps,
            compute_kernel_config=self.config.compute_kernel_cfg,
            memory_config=self.config.output_memcfg,
        )


def _resolve(config: RMSNormConfig) -> RMSNormConfig:
    device = config.mesh_device or config.weight.device or ttnn.GetDefaultDevice()
    if device is None:
        raise ValueError("PplxRMSNorm needs a device")
    return replace(
        config,
        mesh_device=device,
        weight=resolve(config.weight, device),
        compute_kernel_cfg=config.compute_kernel_cfg
        or ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        ),
        output_memcfg=config.output_memcfg or ttnn.DRAM_MEMORY_CONFIG,
    )
