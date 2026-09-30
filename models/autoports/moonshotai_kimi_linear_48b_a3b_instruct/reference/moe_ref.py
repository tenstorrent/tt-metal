# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sigmoid-router MoE (256 experts, top-8, bias-corrected choice, renormalised, scaled) + shared expert; dense SwiGLU."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, w_gate.float())) * F.linear(x, w_up.float()), w_down.float())


def dense_mlp_reference(hidden: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return swiglu(
        hidden.float(), weights["mlp.gate_proj.weight"], weights["mlp.up_proj.weight"], weights["mlp.down_proj.weight"]
    )


def router_reference(
    hidden: torch.Tensor, gate_weight: torch.Tensor, bias: torch.Tensor, cfg: KimiLinearConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (topk_idx [N,k] int64, topk_weight [N,k] fp32, dense routing [N,E] fp32 with weights at chosen experts)."""
    x = hidden.float().reshape(-1, hidden.shape[-1])
    logits = F.linear(x, gate_weight.float())
    scores = torch.sigmoid(logits)
    choice = scores + bias.float()
    idx = torch.topk(choice, cfg.num_experts_per_token, dim=-1, sorted=False).indices
    w = scores.gather(-1, idx)
    if cfg.moe_renormalize:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * cfg.routed_scaling_factor
    dense = torch.zeros_like(scores).scatter(-1, idx, w)
    return idx, w, dense


def moe_reference(hidden: torch.Tensor, weights: Mapping[str, torch.Tensor], cfg: KimiLinearConfig) -> torch.Tensor:
    """hidden [B,T,H] -> [B,T,H] fp32 (routed experts + shared expert). Expert stacks are [E,out,in] torch layout."""
    shape = hidden.shape
    x = hidden.float().reshape(-1, shape[-1])
    _, _, dense = router_reference(x, weights["moe.gate.weight"], weights["moe.gate.e_score_correction_bias"], cfg)
    g, u, d = (
        weights["moe.experts.gate"].float(),
        weights["moe.experts.up"].float(),
        weights["moe.experts.down"].float(),
    )
    out = torch.zeros_like(x)
    for e in torch.nonzero(dense.sum(0) > 0).flatten().tolist():
        rows = torch.nonzero(dense[:, e] > 0).flatten()
        y = swiglu(x[rows], g[e], u[e], d[e])
        out.index_add_(0, rows, y * dense[rows, e].unsqueeze(-1))
    if cfg.num_shared_experts:
        out = out + swiglu(
            x,
            weights["moe.shared.gate_proj.weight"],
            weights["moe.shared.up_proj.weight"],
            weights["moe.shared.down_proj.weight"],
        )
    return out.reshape(shape)
