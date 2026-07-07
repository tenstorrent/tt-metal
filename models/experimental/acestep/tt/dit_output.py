# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 DiT output head: norm_out (AdaLN) + proj_out (de-patchify).

Reference: tail of AceStepDiTModel.forward:

    shift, scale = (scale_shift_table[1,2,dim] + temb.unsqueeze(1)).chunk(2, dim=1)
    x = norm_out(x) * (1 + scale) + shift          # 2-value AdaLN
    x = proj_out(x)                                # ConvTranspose1d de-patchify
    x = x[:, :original_seq_len, :]                 # crop padding (caller's responsibility)

proj_out = ConvTranspose1d(in=2048, out=audio_acoustic_hidden_dim=64, kernel=stride=p=2).
Because kernel==stride, it is an un-patchify: Linear(in -> out*p) in (p, out) order, then reshape
[B, T', out*p] -> [B, T'*p, out]. Verified PCC ~1.0 vs the real ConvTranspose1d. We fold the conv
into a Linear on host (no device conv needed).

Reuses RMSNorm1D for norm_out; modulation + de-patchify via plain ttnn ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig


@dataclass
class DiTOutputConfig:
    # scale_shift_table [1, 2, dim] LazyWeight.
    scale_shift_table: LazyWeight
    # norm_out RMSNorm weight.
    norm_out_weight: LazyWeight
    # proj_out folded linear [in, out*p] (transposed for ttnn.linear) + bias [1, out].
    proj_out_weight: LazyWeight
    proj_out_bias: LazyWeight

    dim: int = 2048
    out_channels: int = 64
    patch_size: int = 2
    eps: float = 1e-6
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "DiTOutputConfig":
        if self.mesh_device is None:
            self.mesh_device = self.norm_out_weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self


class DiTOutput(LightweightModule):
    """forward(x [1,1,T',dim], temb [1,1,B,dim]) -> [1,1,T'*p, out_channels]."""

    def __init__(self, config: DiTOutputConfig):
        self.config = config.resolved()
        cfg = self.config
        self.scale_shift_table = cfg.scale_shift_table.get_device_weight()  # [1,2,dim]
        self.norm_out = RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.norm_out_weight, eps=cfg.eps))
        self.proj_weight = cfg.proj_out_weight.get_device_weight()
        self.proj_bias = cfg.proj_out_bias.get_device_weight()

    @classmethod
    def from_config(cls, config: DiTOutputConfig):
        return cls(config)

    def forward(self, x, temb):
        cfg = self.config
        # Modulation: scale_shift_table[1,2,dim] + temb[1,1,B,dim].unsqueeze -> [1,2,B,dim] chunk(2).
        sst = ttnn.reshape(self.scale_shift_table, (1, 2, 1, cfg.dim))
        mod = ttnn.add(sst, temb)  # broadcast temb [1,1,B,dim] over the 2-axis
        shift, scale = ttnn.chunk(mod, 2, dim=1)  # one chunk vs 2 slices

        x = self.norm_out.forward(x, mode="prefill")
        x = ttnn.mac(x, ttnn.add(scale, 1.0), shift)  # norm*(1+scale)+shift, fused mul+add

        # De-patchify: Linear(dim -> out*p) then reshape [1,1,T',out*p] -> [1,1,T'*p, out].
        # ConvTranspose1d bias is per-output-channel (out), applied AFTER un-patchify (each of
        # the p positions shares the same out-channel bias) -> cannot fold into the linear.
        y = ttnn.linear(x, self.proj_weight, compute_kernel_config=cfg.compute_kernel_config)
        tprime = y.shape[2]
        p, out = cfg.patch_size, cfg.out_channels
        y = ttnn.reshape(y, (1, 1, tprime * p, out))
        y = ttnn.add(y, self.proj_bias)  # broadcast [1,out] over [1,1,T'*p,out]
        return y
