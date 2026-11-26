# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP with the GLM-5.3 clamps (dense MLP of layers 0-2, 12288; shared expert, 2048).

    y = W_down( SiLU(min(x W_gate, limit)) * clamp(x W_up, -limit, limit) ),  limit = swiglu_limit = 10   (modeling:97-103)

gate and up are fused into one bfloat8_b weight `[4096, 2I]` (columns [gate | up]) at load time.
"""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi2_fp32

DRAM = ttnn.DRAM_MEMORY_CONFIG


def swiglu_clamped(gate_up: ttnn.Tensor, intermediate: int, limit: float) -> ttnn.Tensor:
    """gate_up [..., 2I] = [gate | up] -> SiLU(min(gate, limit)) * clamp(up, -limit, limit) [..., I]."""
    shape = list(gate_up.shape)
    begin = [0] * len(shape)
    gate = ttnn.slice(gate_up, begin, shape[:-1] + [intermediate])
    up = ttnn.slice(gate_up, begin[:-1] + [intermediate], shape)
    gate = ttnn.clamp(gate, max=limit)
    up = ttnn.clamp(up, min=-limit, max=limit)
    return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])


class SwiGLUMLP:
    def __init__(self, mesh_device, gate_t: ttnn.Tensor, up_t: ttnn.Tensor, down_t: ttnn.Tensor, *, limit: float):
        """gate_t/up_t [4096, I], down_t [I, 4096] bfloat8_b (cache layout [in, out])."""
        self.intermediate = gate_t.shape[-1]
        self.limit = limit
        self.gate_up = ttnn.concat([gate_t, up_t], dim=-1, memory_config=DRAM)
        ttnn.deallocate(gate_t)
        ttnn.deallocate(up_t)
        self.down = down_t
        self.ckc = hifi2_fp32(mesh_device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1, 1, T, 4096] bf16 -> [1, 1, T, 4096] bf16."""
        gu = ttnn.linear(x, self.gate_up, compute_kernel_config=self.ckc, memory_config=DRAM, dtype=ttnn.bfloat16)
        act = swiglu_clamped(gu, self.intermediate, self.limit)
        ttnn.deallocate(gu)
        return ttnn.linear(act, self.down, compute_kernel_config=self.ckc, memory_config=DRAM, dtype=ttnn.bfloat16)
