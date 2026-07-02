# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 TimestepEmbedding (TTTv2-pattern custom module).

Reference: TimestepEmbedding in modeling_acestep_v15_base.py.
  sinusoidal(t) -> linear_1 -> SiLU -> linear_2 = temb  [B, time_embed_dim]
  time_proj(SiLU(temb)) -> unflatten(6) = timestep_proj [B, 6, time_embed_dim]

The sinusoidal timestep embedding is a small, data-dependent host computation (a table of
cos/sin over the raw timestep scalar) — cheap and exact on host. The two Linears + time_proj
(the parameterized, compute-heavy part) run on device via ttnn ops, matching how DiT stacks
feed `timestep_proj` into AdaLN modulation.

No TTTv2 library module matches this shape (it's a diffusion-specific conditioning head), so
we follow the pattern: LightweightModule + Config + from_config + straight-line forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight


@dataclass
class TimestepEmbeddingConfig:
    # Linear weights already transposed to [in, out] for ttnn.linear; biases as [1, out].
    linear_1_weight: LazyWeight
    linear_1_bias: LazyWeight
    linear_2_weight: LazyWeight
    linear_2_bias: LazyWeight
    time_proj_weight: LazyWeight
    time_proj_bias: LazyWeight

    in_channels: int = 256
    time_embed_dim: int = 2048
    scale: float = 1000.0
    max_period: int = 10000

    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "TimestepEmbeddingConfig":
        if self.mesh_device is None:
            self.mesh_device = self.linear_1_weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self


class TimestepEmbedding(LightweightModule):
    """forward(t: torch.Tensor [B]) -> (temb [1,1,B,dim], timestep_proj [1,6,B,dim])."""

    def __init__(self, config: TimestepEmbeddingConfig):
        self.config = config.resolved()
        cfg = self.config
        self.w1 = cfg.linear_1_weight.get_device_weight()
        self.b1 = cfg.linear_1_bias.get_device_weight()
        self.w2 = cfg.linear_2_weight.get_device_weight()
        self.b2 = cfg.linear_2_bias.get_device_weight()
        self.wp = cfg.time_proj_weight.get_device_weight()
        self.bp = cfg.time_proj_bias.get_device_weight()

    @classmethod
    def from_config(cls, config: TimestepEmbeddingConfig):
        return cls(config)

    def _sinusoid(self, t: torch.Tensor) -> torch.Tensor:
        """Host sinusoidal embedding, matching the reference exactly. t:[B] -> [B, in_channels]."""
        cfg = self.config
        dim = cfg.in_channels
        t = t * cfg.scale
        half = dim // 2
        freqs = torch.exp(-math.log(cfg.max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # [B, in_channels]

    def forward(self, t: torch.Tensor):
        cfg = self.config
        ck = cfg.compute_kernel_config
        b = t.shape[0]

        t_freq = self._sinusoid(t)  # host [B, in_channels]
        x = ttnn.from_torch(
            t_freq.reshape(1, 1, b, cfg.in_channels),
            device=cfg.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        temb = ttnn.linear(x, self.w1, bias=self.b1, compute_kernel_config=ck)
        temb = ttnn.silu(temb)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, compute_kernel_config=ck)  # [1,1,B,dim]

        proj_in = ttnn.silu(temb)
        timestep_proj = ttnn.linear(proj_in, self.wp, bias=self.bp, compute_kernel_config=ck)  # [1,1,B,6*dim]
        timestep_proj = ttnn.reshape(timestep_proj, (1, b, 6, cfg.time_embed_dim))
        timestep_proj = ttnn.transpose(timestep_proj, 1, 2)  # [1,6,B,dim]
        return temb, timestep_proj
