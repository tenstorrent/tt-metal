# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP with the GLM-5.3-Flash clamp (``swiglu_limit``).

HF ``Glm5NextTextMLP``:
    gate = clamp(x @ W_gate^T, max=limit); up = clamp(x @ W_up^T, -limit, limit)
    y = (silu(gate) * up) @ W_down^T

Used for the three dense layers (intermediate 12288) and as the shared expert of every MoE
layer (intermediate 2048). Single-device, interleaved DRAM, tile layout; the input is the
sublayer activation ``[1, 1, T, hidden]`` (prefill, T tokens) or ``[1, 1, B, hidden]`` (decode).
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _compute_config(device, fidelity=ttnn.MathFidelity.HiFi2):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


class TtClampedSwiGLU(LightweightModule):
    def __init__(
        self,
        device,
        gate_proj: torch.Tensor,  # [I, H] (HF nn.Linear layout)
        up_proj: torch.Tensor,  # [I, H]
        down_proj: torch.Tensor,  # [H, I]
        swiglu_limit: float,
        *,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.limit = float(swiglu_limit)
        self.activation_dtype = activation_dtype
        self.hidden = gate_proj.shape[1]
        self.intermediate = gate_proj.shape[0]
        assert up_proj.shape == gate_proj.shape and tuple(down_proj.shape) == (self.hidden, self.intermediate)
        # fused [H, 2I]: gate columns first, then up
        w13 = torch.cat([gate_proj.t(), up_proj.t()], dim=-1).contiguous()
        self.w13 = ttnn.from_torch(w13, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.w2 = ttnn.from_torch(
            down_proj.t().contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.compute = _compute_config(device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, T, H] -> [1, 1, T, H]."""
        T = x.shape[-2]
        gu = ttnn.linear(x, self.w13, dtype=self.activation_dtype, compute_kernel_config=self.compute)
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, T, self.intermediate])
        up = ttnn.slice(gu, [0, 0, 0, self.intermediate], [1, 1, T, 2 * self.intermediate])
        ttnn.deallocate(gu)
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        act = ttnn.silu(gate)
        ttnn.deallocate(gate)
        h = ttnn.multiply(act, up, output_tensor=act)
        ttnn.deallocate(up)
        y = ttnn.linear(h, self.w2, dtype=self.activation_dtype, compute_kernel_config=self.compute)
        ttnn.deallocate(h)
        return y
