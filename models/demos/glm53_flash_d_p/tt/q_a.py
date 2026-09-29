# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 DSA q_a stem on device, replicated: q_resid = q_a_layernorm(x @ q_a_proj^T), [S, 4096] -> [S, 1536].

q_a_proj: fp8 e4m3 + 128x128 block scale, dequantized by the reference loader, stored bf16 [H, q_lora_rank].
The projection runs HiFi4 + fp32 acc into fp32; the norm (tt/rms_norm.py:TtRMSNorm, ttnn.bringup.rms_norm) runs on the
fp32 projection over all 1536 columns (eps 1e-5), then the result is cast to bf16. No CCL (every chip holds all rows).
From models/demos/deepseek_v3_d_p/tt/mla/mla.py (q_a_proj -> q_a_layernorm).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate
from models.demos.glm53_flash_d_p.tt.rms_norm import TtRMSNorm


class TtQA:
    def __init__(self, mesh, w_q_a: torch.Tensor, norm_w: torch.Tensor, eps: float = 1e-5):
        """w_q_a: HF [q_lora_rank, H] (dequantized); norm_w: [q_lora_rank]."""
        rank, hidden = w_q_a.shape
        self.w = replicate(mesh, w_q_a.float().T.reshape(1, 1, hidden, rank).to(torch.bfloat16))
        self.norm = TtRMSNorm(mesh, norm_w, eps)
        self.cfg = hifi4_config()

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE bf16 (attn_norm output). Returns replicated [1, 1, S, q_lora_rank] bf16."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        p = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=mc)
        y = self.norm(p)
        ttnn.deallocate(p)
        out = ttnn.typecast(y, ttnn.bfloat16, memory_config=mc)
        ttnn.deallocate(y)
        return out


def build_q_a(mesh, loader, cfg, layer: int) -> TtQA:
    p = f"{PREFIX}layers.{layer}.self_attn."
    return TtQA(
        mesh,
        loader.weight(p + "q_a_proj.weight"),
        loader.weight(p + "q_a_layernorm.weight"),
        cfg.rms_norm_eps,
    )
