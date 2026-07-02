# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 DiT patch embedding (proj_in) and de-patchify (proj_out).

Reference: AceStepDiTModel.proj_in / proj_out in modeling_acestep_v15_base.py.

proj_in  = Conv1d(in_channels=192, out=2048, kernel=stride=patch_size=2). Because kernel==stride,
this is exactly a *patchify*: reshape [B, T, C] -> [B, T/p, C*p] (channel-major within patch)
then a Linear(C*p -> out). We use that equivalence (verified PCC 1.0) — cleaner and more accurate
than a strided conv on device.

proj_out = ConvTranspose1d(in=2048, out=audio_acoustic_hidden_dim=64, kernel=stride=p). Symmetric:
Linear(in -> out*p) then un-patchify [B, T', out*p] -> [B, T'*p, out].

Both run as ttnn.linear over a host-side reshape. The Conv weight is folded into a Linear weight
on load (host), so no conv kernel is needed on device.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight


@dataclass
class PatchEmbedConfig:
    # Folded linear weight [in*p, out] (already transposed for ttnn.linear) + bias [1, out].
    weight: LazyWeight
    bias: LazyWeight
    in_channels: int = 192
    out_channels: int = 2048
    patch_size: int = 2
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "PatchEmbedConfig":
        if self.mesh_device is None:
            self.mesh_device = self.weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        return self


class PatchEmbed(LightweightModule):
    """proj_in: forward(x [1,1,T,C]) -> [1,1,T/p, out]. T must be divisible by patch_size."""

    def __init__(self, config: PatchEmbedConfig):
        self.config = config.resolved()
        self.weight = self.config.weight.get_device_weight()
        self.bias = self.config.bias.get_device_weight()

    @classmethod
    def from_config(cls, config: PatchEmbedConfig):
        return cls(config)

    def forward(self, x):
        cfg = self.config
        p, c = cfg.patch_size, cfg.in_channels
        t = x.shape[2]
        # [1,1,T,C] -> [1, T/p, p, C] -> [1, T/p, C, p] -> [1,1,T/p, C*p] (channel-major within patch).
        x = ttnn.reshape(x, (1, t // p, p, c))
        x = ttnn.permute(x, (0, 1, 3, 2))  # [1, T/p, C, p]
        x = ttnn.reshape(x, (1, 1, t // p, c * p))
        return ttnn.linear(x, self.weight, bias=self.bias, compute_kernel_config=cfg.compute_kernel_config)
