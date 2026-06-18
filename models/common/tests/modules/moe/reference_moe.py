# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Parameterized torch golden reference for the TTTv2 MoE1D module.

This is the PCC oracle for ``models/common/modules/moe/moe_1d.py``. It is a plain,
dense, loop-over-experts implementation (Mixtral/Qwen-style) with knobs that let a
single reference reproduce **both** families MoE1D must match:

  - **Gemma4**: GeGLU activation, no projection bias, router pre-RMSNorm + scales,
    expert pre-RMSNorm, ``softmax -> topk -> renormalize`` routing, optional
    per-expert output scale.
  - **GPT-OSS**: clamped-SwiGLU (alpha, limit), bias on every projection, no router
    pre-norm (caller pre-normed), ``topk -> softmax`` routing.

Weights are kept in **HuggingFace layout** (un-transposed):
  - ``router_weight``  : [E, H]            logits = x @ router_weight.T
  - ``gate_proj``      : [E, I, H]         g = x @ gate_proj[e].T
  - ``up_proj``        : [E, I, H]         u = x @ up_proj[e].T
  - ``down_proj``      : [E, H, I]         o = a @ down_proj[e].T
The device module stores these transposed for matmul; the test harness performs the
transpose when building ``LazyWeight``s, so this reference stays in the natural HF
layout used by the source state_dicts.

torch-only on purpose (no ttnn / no enums imported from the module) so it can run on a
plain CPU box and serve as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

# Activation strategy strings (mirror MoE1D's ExpertActivation enum values).
GEGLU = "geglu"
SWIGLU = "swiglu"  # plain SwiGLU: silu(gate) * up (Qwen3-MoE / Granite-4 / North-Mini).
CLAMPED_SWIGLU = "clamped_swiglu"

# Routing-normalization strategy strings (mirror MoE1D's RoutingNorm enum values).
SOFTMAX_TOPK_RENORM = "softmax_topk_renorm"  # Gemma / Qwen3: softmax over all experts, topk, then renormalize topk.
TOPK_SOFTMAX = "topk_softmax"  # GPT-OSS / Granite-4: topk over raw logits, then softmax over the topk.
TOPK_SIGMOID = "topk_sigmoid"  # North-Mini: topk over raw logits, then per-expert sigmoid, no renorm.


@dataclass
class MoEReferenceWeights:
    """Per-MoE weight tensors in HF layout. Biases / norm weights optional."""

    router_weight: torch.Tensor  # [E, H]
    gate_proj: torch.Tensor  # [E, I, H]
    up_proj: torch.Tensor  # [E, I, H]
    down_proj: torch.Tensor  # [E, H, I]

    router_bias: Optional[torch.Tensor] = None  # [E]
    gate_bias: Optional[torch.Tensor] = None  # [E, I]
    up_bias: Optional[torch.Tensor] = None  # [E, I]
    down_bias: Optional[torch.Tensor] = None  # [E, H]

    # Router pre-processing (Gemma). All optional; None => step skipped.
    router_norm_weight: Optional[torch.Tensor] = None  # [H]  RMSNorm gamma on router input
    router_input_scale: Optional[torch.Tensor] = None  # [H]  elementwise scale after router RMSNorm
    per_expert_scale: Optional[torch.Tensor] = None  # [E]  scale on dense routing after scatter

    # Expert pre-processing (Gemma pre_feedforward_layernorm_2). None => experts use x directly.
    expert_norm_weight: Optional[torch.Tensor] = None  # [H]  RMSNorm gamma on expert input


@dataclass
class MoEReferenceConfig:
    num_experts: int
    top_k: int
    expert_activation: str = GEGLU
    routing_norm: str = SOFTMAX_TOPK_RENORM
    # GPT-OSS clamped-SwiGLU params (ignored for GEGLU).
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    # Gemma router logit scale (multiply router input by hidden_size**-0.5). None => skip.
    router_logit_scale: Optional[float] = None
    rms_norm_eps: float = 1e-6
    # Match HF gemma which upcasts router math to fp32; keeps the golden numerically tight.
    router_in_fp32: bool = True
    # Emulate the device's bf16 topk: round the topk-selection input to bf16 before topk so
    # the selected expert set matches the hardware (which only supports bf16/bf8 topk).
    route_topk_in_bf16: bool = False


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma-style RMSNorm: x / rms(x) * (1 + 0)·weight. Uses fp32 for the reduction.

    NOTE: Gemma applies ``weight`` directly (not ``1 + weight``) inside the router
    (see Gemma4Router which loads ``scale`` separately). We keep the plain
    ``weight`` form here; if a reference uses the ``(1 + weight)`` convention, pass
    the pre-incremented gamma.
    """
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    out = xf * weight.float()
    return out.to(dtype)


def _expert_mlp(
    x: torch.Tensor,  # [T, H]  tokens routed to one expert (or all tokens, dense)
    e: int,
    w: MoEReferenceWeights,
    cfg: MoEReferenceConfig,
) -> torch.Tensor:
    """Single-expert gate/up -> activation -> down. Returns [T, H]."""
    g = F.linear(x, w.gate_proj[e], None if w.gate_bias is None else w.gate_bias[e])  # [T, I]
    u = F.linear(x, w.up_proj[e], None if w.up_bias is None else w.up_bias[e])  # [T, I]

    if cfg.expert_activation == GEGLU:
        # Gemma: gelu_pytorch_tanh(gate) * up
        act = F.gelu(g, approximate="tanh") * u
    elif cfg.expert_activation == SWIGLU:
        # Plain SwiGLU (Qwen3-MoE / Granite-4 / North-Mini): silu(gate) * up
        act = F.silu(g) * u
    elif cfg.expert_activation == CLAMPED_SWIGLU:
        # GPT-OSS: g=clamp(g, max=limit); u=clamp(u, -limit, limit);
        #          glu = g * sigmoid(alpha*g); out = (u + 1) * glu
        limit = cfg.swiglu_limit
        gc = g.clamp(max=limit)
        uc = u.clamp(min=-limit, max=limit)
        glu = gc * torch.sigmoid(cfg.swiglu_alpha * gc)
        act = (uc + 1.0) * glu
    else:
        raise ValueError(f"unknown expert_activation: {cfg.expert_activation}")

    o = F.linear(act, w.down_proj[e], None if w.down_bias is None else w.down_bias[e])  # [T, H]
    return o


def _compute_dense_routing(
    router_in: torch.Tensor,  # [T, H]
    w: MoEReferenceWeights,
    cfg: MoEReferenceConfig,
) -> torch.Tensor:
    """Returns dense routing weights [T, E] with zeros at non-selected experts.

    Mirrors MoE1D's router: optional pre-norm/scales -> linear -> routing-norm
    strategy -> scatter to dense -> optional per_expert_scale.
    """
    x = router_in
    if w.router_norm_weight is not None:
        x = _rmsnorm(x, w.router_norm_weight, cfg.rms_norm_eps)
    if w.router_input_scale is not None:
        x = x * w.router_input_scale
    if cfg.router_logit_scale is not None:
        x = x * cfg.router_logit_scale

    work_dtype = torch.float32 if cfg.router_in_fp32 else x.dtype
    logits = F.linear(
        x.to(work_dtype),
        w.router_weight.to(work_dtype),
        None if w.router_bias is None else w.router_bias.to(work_dtype),
    )  # [T, E]

    def _bf16(t):
        return t.to(torch.bfloat16).float() if cfg.route_topk_in_bf16 else t

    if cfg.routing_norm == SOFTMAX_TOPK_RENORM:
        probs = F.softmax(logits, dim=-1)
        _, top_i = torch.topk(_bf16(probs), cfg.top_k, dim=-1)  # selection on bf16-rounded probs
        top_v = torch.gather(probs, -1, top_i)  # gather fp32 probs at the selected experts
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    elif cfg.routing_norm == TOPK_SOFTMAX:
        _, top_i = torch.topk(_bf16(logits), cfg.top_k, dim=-1)
        top_v = torch.gather(logits, -1, top_i)
        top_v = F.softmax(top_v, dim=-1)
    elif cfg.routing_norm == TOPK_SIGMOID:
        # North-Mini: independent per-expert gate, no renorm.
        _, top_i = torch.topk(_bf16(logits), cfg.top_k, dim=-1)
        top_v = torch.gather(logits, -1, top_i)
        top_v = torch.sigmoid(top_v)
    else:
        raise ValueError(f"unknown routing_norm: {cfg.routing_norm}")

    dense = torch.zeros_like(logits)
    dense.scatter_(dim=-1, index=top_i, src=top_v)

    if w.per_expert_scale is not None:
        dense = dense * w.per_expert_scale.to(dense.dtype)

    return dense


def reference_moe_forward(
    x: torch.Tensor,  # [B, S, H] or [T, H]
    w: MoEReferenceWeights,
    cfg: MoEReferenceConfig,
) -> torch.Tensor:
    """Dense golden MoE forward. Returns same leading shape as ``x``, last dim H.

    Steps:
      1. router routing weights from ``x`` (router owns its optional pre-norm).
      2. expert input = optional expert RMSNorm(x), else x.
      3. for every expert: run dense MLP over all tokens, scale by routing[:, e], sum.
    (Computing all experts densely matches the device prefill path and is exact for
    decode too — non-selected experts contribute 0 via the dense routing weights.)
    """
    orig_shape = x.shape
    H = orig_shape[-1]
    x2d = x.reshape(-1, H)  # [T, H]
    T = x2d.shape[0]

    dense_routing = _compute_dense_routing(x2d, w, cfg)  # [T, E]

    expert_in = x2d
    if w.expert_norm_weight is not None:
        expert_in = _rmsnorm(x2d, w.expert_norm_weight, cfg.rms_norm_eps)

    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(cfg.num_experts):
        weight_e = dense_routing[:, e]  # [T]
        # dense compute then mask by routing weight (selected tokens have nonzero weight)
        o_e = _expert_mlp(expert_in, e, w, cfg).float()  # [T, H]
        out = out + weight_e.unsqueeze(-1).float() * o_e

    return out.to(x.dtype).reshape(*orig_shape[:-1], H)


# ---------------------------------------------------------------------------
# Convenience preset builders — used by tests to spin up each family's config.
# ---------------------------------------------------------------------------


def gemma_preset(num_experts: int, top_k: int, hidden_size: int) -> MoEReferenceConfig:
    return MoEReferenceConfig(
        num_experts=num_experts,
        top_k=top_k,
        expert_activation=GEGLU,
        routing_norm=SOFTMAX_TOPK_RENORM,
        router_logit_scale=hidden_size**-0.5,
    )


def gpt_oss_preset(num_experts: int, top_k: int, alpha: float = 1.702, limit: float = 7.0) -> MoEReferenceConfig:
    return MoEReferenceConfig(
        num_experts=num_experts,
        top_k=top_k,
        expert_activation=CLAMPED_SWIGLU,
        routing_norm=TOPK_SOFTMAX,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
    )


def qwen3_preset(num_experts: int, top_k: int) -> MoEReferenceConfig:
    """Qwen3-30B-A3B: plain SwiGLU, softmax->topk->renorm (norm_topk_prob=True), no bias/norm."""
    return MoEReferenceConfig(
        num_experts=num_experts,
        top_k=top_k,
        expert_activation=SWIGLU,
        routing_norm=SOFTMAX_TOPK_RENORM,
    )


def granite_preset(num_experts: int, top_k: int) -> MoEReferenceConfig:
    """Granite-4-H-Tiny routed-expert block: plain SwiGLU, topk->softmax, no bias/norm."""
    return MoEReferenceConfig(
        num_experts=num_experts,
        top_k=top_k,
        expert_activation=SWIGLU,
        routing_norm=TOPK_SOFTMAX,
    )


def north_mini_preset(num_experts: int, top_k: int) -> MoEReferenceConfig:
    """North-Mini-Code-1.0: plain SwiGLU, topk->sigmoid (no renorm), no bias/norm."""
    return MoEReferenceConfig(
        num_experts=num_experts,
        top_k=top_k,
        expert_activation=SWIGLU,
        routing_norm=TOPK_SIGMOID,
    )
