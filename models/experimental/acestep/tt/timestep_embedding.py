# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 TimestepEmbedding (TTTv2-pattern custom module).

Reference: TimestepEmbedding in modeling_acestep_v15_base.py.
  sinusoidal(t) -> linear_1 -> SiLU -> linear_2 = temb  [B, time_embed_dim]
  time_proj(SiLU(temb)) -> unflatten(6) = timestep_proj [B, 6, time_embed_dim]

The sinusoidal timestep embedding runs **entirely on device** (ttnn): a constant frequency table
(`time_proj_factor`) is built once at construction (host work, outside any trace region) and held
as a resident ttnn buffer; each forward does `mul -> cos/sin -> concat` on-device. This makes the
whole module trace-capturable — no per-step `torch`/`from_torch` inside forward. Pattern mirrors
tt_dit `layers/embeddings.py::Timesteps`. The two Linears + time_proj run on device as before.

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
    dtype: ttnn.DataType = ttnn.bfloat16

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
        # Constant sinusoid frequency table, built ONCE (host) and held resident on device so
        # forward() is pure-ttnn (trace-safe). Shape [1,1,1,half] to broadcast against t [1,1,B,1].
        # reference: args = (t * scale) * exp(-log(max_period) * arange(half) / half). We fold the
        # scalar `scale` INTO the table so both input paths (torch or ttnn) feed RAW t and get
        # identical scaling -> args = t * freq_table.
        # fp32 for the sinusoid: args = t*scale*freqs reaches ~1000, and cos/sin of large arguments
        # in bf16 (~3 sig digits) loses precision (drops PCC to ~0.97). The reference is fp32; match
        # it here, then cast to bf16 only for the matmuls.
        half = cfg.in_channels // 2
        freqs = torch.exp(-math.log(cfg.max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.freq_table = ttnn.from_torch(
            (freqs * cfg.scale).reshape(1, 1, 1, half),
            device=cfg.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

    @classmethod
    def from_config(cls, config: TimestepEmbeddingConfig):
        return cls(config)

    def _to_device_t(self, t) -> ttnn.Tensor:
        """Normalize the timestep input to a device tensor [1,1,B,1].

        Accepts a torch tensor [B] (eager/test path — the from_torch here is fine because it is the
        INPUT boundary, done before any trace region), or an already-on-device ttnn tensor [1,1,B,1]
        (traced path — the resident timestep buffer, no host work).
        """
        if isinstance(t, ttnn.Tensor):
            return t
        b = t.shape[0]
        return ttnn.from_torch(
            t.reshape(1, 1, b, 1).float(),  # RAW t; scale is folded into freq_table
            device=self.config.mesh_device,
            dtype=ttnn.float32,  # fp32 sinusoid (see freq_table note)
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, t):
        cfg = self.config
        ck = cfg.compute_kernel_config

        # On-device sinusoid: t[1,1,B,1] * freq_table[1,1,1,half] -> [1,1,B,half]; cos|sin concat.
        # reference order is cos-first: cat([cos(args), sin(args)]). in_channels is even (256).
        t_dev = self._to_device_t(t)  # [1,1,B,1] raw t (scale folded into freq_table)
        b = t_dev.shape[2]
        args = ttnn.mul(t_dev, self.freq_table)  # fp32 broadcast -> [1,1,B,half]
        emb = ttnn.concat([ttnn.cos(args), ttnn.sin(args)], dim=-1)  # fp32 [1,1,B,in_channels]
        x = ttnn.typecast(emb, cfg.dtype)  # cast to bf16 for the matmuls
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        temb = ttnn.linear(x, self.w1, bias=self.b1, compute_kernel_config=ck)
        temb = ttnn.silu(temb)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, compute_kernel_config=ck)  # [1,1,B,dim]

        proj_in = ttnn.silu(temb)
        timestep_proj = ttnn.linear(proj_in, self.wp, bias=self.bp, compute_kernel_config=ck)  # [1,1,B,6*dim]
        timestep_proj = ttnn.reshape(timestep_proj, (1, b, 6, cfg.time_embed_dim))
        timestep_proj = ttnn.transpose(timestep_proj, 1, 2)  # [1,6,B,dim]
        return temb, timestep_proj
