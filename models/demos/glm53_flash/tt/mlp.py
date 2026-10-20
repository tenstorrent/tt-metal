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
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, interleave_blocks, single_chip


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
        mesh: GLMMesh | None = None,
    ):
        """Tensor-parallel over the mesh columns (``mesh.tp``): chip ``c`` keeps the ``c``-th slice of
        the intermediate (``[gate_c | up_c]`` of ``w13``, rows ``c`` of ``w2``) and ``forward`` returns
        the all-reduced result (or the TP partial with ``reduce=False``, for callers that fold more
        partials in before one collective)."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.limit = float(swiglu_limit)
        self.activation_dtype = activation_dtype
        self.hidden = gate_proj.shape[1]
        self.intermediate = gate_proj.shape[0]
        assert up_proj.shape == gate_proj.shape and tuple(down_proj.shape) == (self.hidden, self.intermediate)
        tp = self.mesh.tp
        assert self.intermediate % (tp * 32) == 0, (self.intermediate, tp)
        self.intermediate_local = self.intermediate // tp
        # fused [H, 2I] in tp blocks of [gate_c | up_c] so a plain column split hands each chip its pair
        w13 = interleave_blocks([gate_proj.t(), up_proj.t()], tp, dim=-1)
        self.w13 = self.mesh.upload(w13, weight_dtype, cols=-1)
        self.w2 = self.mesh.upload(down_proj.t(), weight_dtype, cols=-2)
        self.compute = _compute_config(device)
        self.lin13 = DecodeLinear(device, self.w13)
        self.lin2 = DecodeLinear(device, self.w2)

    def forward(self, x: ttnn.Tensor, *, reduce: bool = True) -> ttnn.Tensor:
        """x: [1, 1, T, H] (replicated over the TP group) -> [1, 1, T, H] (TP partial if not ``reduce``)."""
        T = x.shape[-2]
        I = self.intermediate_local
        gu = self.lin13(x, dtype=self.activation_dtype, compute_kernel_config=self.compute)
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, T, I])
        up = ttnn.slice(gu, [0, 0, 0, I], [1, 1, T, 2 * I])
        ttnn.deallocate(gu)
        # silu(min(gate, limit)) * clamp(up, -limit, limit) in one op (ttnn.clamped_silu_glu)
        h = ttnn.clamped_silu_glu(gate, up, self.limit)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        y = self.lin2(h, dtype=self.activation_dtype, compute_kernel_config=self.compute)
        ttnn.deallocate(h)
        if reduce:
            y = self.mesh.all_reduce_cols(y)
        return y
