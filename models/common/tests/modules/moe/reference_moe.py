# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Parameterized torch golden for MoE1D.

Implements the union of the Gemma4 and GPT-OSS router + routed-experts math, parameterized by the
M1 variation matrix (routing strategy, activation, biases, scales). Used as the clean-room math
oracle for the MoE1D unit tests, and to cross-check that MoE1D reproduces each reference's output.

Validate against the references (selection agreement) before trusting as the parity target — see
`test_moe_1d.py::test_reference_moe_*` cross-checks and MODULE_NOTES.md.
"""

from dataclasses import dataclass

import torch


@dataclass
class RefMoEConfig:
    top_k: int
    routing_strategy: str  # "softmax_topk_sumnorm" (gemma4) | "topk_softmax" (gpt_oss)
    activation: str  # "geglu" (gemma4) | "swiglu_clamp" (gpt_oss)
    # optional router pre-linear transforms
    router_prenorm_eps: float | None = None
    router_input_scalar: float | None = None
    # activation params (swiglu only)
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalize with no affine weight (matches ttnn.rms_norm without weight)."""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def reference_router(
    router_input: torch.Tensor,  # [..., S, H]
    router_weight: torch.Tensor,  # [H, E]  (ttnn linear layout)
    cfg: RefMoEConfig,
    router_scale: torch.Tensor | None = None,  # [..., H] broadcastable
    router_bias: torch.Tensor | None = None,  # [E]
    per_expert_scale: torch.Tensor | None = None,  # [E]
) -> torch.Tensor:
    """Return dense routing weights [..., S, E] (0 at unselected experts)."""
    x = router_input.float()
    if cfg.router_prenorm_eps is not None:
        x = _rms_normalize(x, cfg.router_prenorm_eps)
    if router_scale is not None:
        x = x * router_scale.float()
    if cfg.router_input_scalar is not None:
        x = x * cfg.router_input_scalar

    logits = x @ router_weight.float()
    if router_bias is not None:
        logits = logits + router_bias.float()

    E = logits.shape[-1]
    if cfg.routing_strategy == "softmax_topk_sumnorm":
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, cfg.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
    elif cfg.routing_strategy == "topk_softmax":
        top_vals, top_idx = torch.topk(logits, cfg.top_k, dim=-1, sorted=True)
        top_vals = torch.softmax(top_vals, dim=-1)
    else:
        raise ValueError(f"unknown routing_strategy {cfg.routing_strategy}")

    dense = torch.zeros_like(logits)
    dense.scatter_(-1, top_idx, top_vals)
    if per_expert_scale is not None:
        dense = dense * per_expert_scale.float()
    return dense


def _activation(gate: torch.Tensor, up: torch.Tensor, cfg: RefMoEConfig) -> torch.Tensor:
    if cfg.activation == "geglu":
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    elif cfg.activation == "swiglu_clamp":
        gate = gate.clamp(max=cfg.swiglu_limit)
        up = up.clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        glu = gate * torch.sigmoid(cfg.swiglu_alpha * gate)
        return (up + 1.0) * glu
    else:
        raise ValueError(f"unknown activation {cfg.activation}")


def reference_experts(
    expert_input: torch.Tensor,  # [..., S, H]
    gate_proj: torch.Tensor,  # [E, H, I]
    up_proj: torch.Tensor,  # [E, H, I]
    down_proj: torch.Tensor,  # [E, I, H]
    dense_routing: torch.Tensor,  # [..., S, E]
    cfg: RefMoEConfig,
    gate_bias: torch.Tensor | None = None,  # [E, I]
    up_bias: torch.Tensor | None = None,  # [E, I]
    down_bias: torch.Tensor | None = None,  # [E, H]
) -> torch.Tensor:
    """Return [..., S, H]: weighted sum over experts of the routed expert MLP outputs."""
    x = expert_input.float().reshape(-1, expert_input.shape[-1])  # [S, H]
    routing = dense_routing.float().reshape(-1, dense_routing.shape[-1])  # [S, E]

    gate = torch.einsum("sh,ehi->sei", x, gate_proj.float())
    up = torch.einsum("sh,ehi->sei", x, up_proj.float())
    if gate_bias is not None:
        gate = gate + gate_bias.float().unsqueeze(0)
    if up_bias is not None:
        up = up + up_bias.float().unsqueeze(0)

    act = _activation(gate, up, cfg)  # [S, E, I]
    down = torch.einsum("sei,eih->seh", act, down_proj.float())  # [S, E, H]
    if down_bias is not None:
        down = down + down_bias.float().unsqueeze(0)

    out = (down * routing.unsqueeze(-1)).sum(dim=1)  # [S, H]
    return out.reshape(expert_input.shape[:-1] + (down.shape[-1],))


def reference_moe(
    router_input: torch.Tensor,
    expert_input: torch.Tensor,
    router_weight: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    cfg: RefMoEConfig,
    router_scale: torch.Tensor | None = None,
    router_bias: torch.Tensor | None = None,
    per_expert_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
):
    """Full MoE golden: router(router_input) -> dense -> experts(expert_input). Returns (out, dense)."""
    dense = reference_router(
        router_input,
        router_weight,
        cfg,
        router_scale=router_scale,
        router_bias=router_bias,
        per_expert_scale=per_expert_scale,
    )
    out = reference_experts(
        expert_input,
        gate_proj,
        up_proj,
        down_proj,
        dense,
        cfg,
        gate_bias=gate_bias,
        up_bias=up_bias,
        down_bias=down_bias,
    )
    return out, dense
