# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi sigmoid router, fully on device and trace-safe.

scores = sigmoid(x @ W^T); choice = scores + e_score_correction_bias; top-8 of choice; weights = scores[chosen] /
sum(scores[chosen]) * routed_scaling_factor. Output: dense routing [1,1,S,E] with the weight at chosen experts, 0 elsewhere
(the sparsity pattern AND the combine weights for sparse_matmul).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor, linear_weight


class KimiRouter:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        *,
        name: str,
        cache_path: Path | None,
    ):
        self.cfg = cfg
        self.top_k = cfg.num_experts_per_token
        self.weight = as_device_tensor(
            mesh_device,
            None if weight is None else linear_weight(weight).reshape(1, 1, cfg.hidden_size, cfg.num_experts),
            name=f"{name}.weight",
            dtype=ttnn.bfloat16,
            shard_dim=None,
            cache_path=cache_path,
        )
        # fp32 end to end: the 8th/9th expert "choice" scores are often closer than one bf16 step (median gap 0.004),
        # so bf16 rounding flips ~9% of the picks vs the HF fp32 router; fp32 logits/sigmoid/bias/topk agree 100%.
        self.bias = as_device_tensor(
            mesh_device,
            None if bias is None else bias.reshape(1, 1, 1, cfg.num_experts).float(),
            name=f"{name}.bias",
            dtype=ttnn.float32,
            shard_dim=None,
            cache_path=cache_path,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,S,H] -> dense routing [1,1,S,E] bf16."""
        logits = ttnn.linear(
            x,
            self.weight,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)
        choice = ttnn.add(scores, self.bias)
        values, idx = ttnn.topk(choice, k=self.top_k, dim=-1)  # fp32 top-k (exact vs the HF fp32 router)
        ttnn.deallocate(choice)
        # scatter does not support fp32 tiled tensors: build the 0/1 selection mask in bf16, then combine in fp32
        ones = ttnn.ones_like(ttnn.typecast(values, ttnn.bfloat16))
        mask = ttnn.scatter(ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16)), dim=-1, index=idx, src=ones)
        ttnn.deallocate(values)
        ttnn.deallocate(idx)
        ttnn.deallocate(ones)
        chosen = ttnn.multiply(scores, ttnn.typecast(mask, ttnn.float32))  # scores at chosen experts, zero elsewhere
        ttnn.deallocate(scores)
        ttnn.deallocate(mask)
        if self.cfg.moe_renormalize:
            denom = ttnn.sum(chosen, dim=-1, keepdim=True)
            chosen = ttnn.div(chosen, denom)
            ttnn.deallocate(denom)
        if self.cfg.routed_scaling_factor != 1.0:
            chosen = ttnn.multiply(chosen, self.cfg.routed_scaling_factor)
        return ttnn.typecast(chosen, ttnn.bfloat16)  # sparse_matmul sparsity / combine weights are bf16
