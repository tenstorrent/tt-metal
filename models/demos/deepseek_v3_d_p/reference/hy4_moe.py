# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for Hy4-preview's FFNs (HF ``hy_v4``), for the module tests of #65890 / #65923.

Same math as the fp32 CPU reference of #65919 (``reference/hy4/model.py``: ``swiglu_mlp`` and ``_moe``), split
into the pieces a module test compares one at a time:

  * ``swiglu_mlp``: HF ``HYV4MLP``, the dense layer-0 FFN (intermediate 18432) and the shared expert (2048).
    NO clamp: ``swiglu_limit`` applies to the routed experts only.
  * ``route``: HF ``HYV4MoE`` router. sigmoid scores; experts are CHOSEN on ``score + e_score_correction_bias``
    (``n_group`` / ``topk_group`` grouping when ``n_group > 1``; Hy4 has 1 / 1, i.e. none) but WEIGHTED by the
    unbiased score, normalised over the top-k and scaled by ``routed_scaling_factor`` (2.827).
  * ``routed_moe``: the routed half, per expert ``silu(min(g, L)) * clamp(u, -L, L)`` with ``L = swiglu_limit``
    (10), on the fused ``gate_up_proj`` ``[E, 2I, H]`` (gate rows first) / ``down_proj`` ``[E, H, I]``.
  * ``clamp_fractions``: how often the clamp actually changes a routed value (gate above L, |up| above L), so a
    test can show its input reaches the clamp.

Everything runs in fp32 on whatever dtype the weights come in. Import-light: torch only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

F32 = torch.float32


def swiglu_mlp(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    """HF ``HYV4MLP`` (dense layer and shared expert): ``(silu(x W_g^T) * (x W_u^T)) W_d^T``, unclamped. Weights in
    HF Linear layout ``[out, in]``."""
    x = x.to(F32)
    return (F.silu(x @ gate_w.to(F32).T) * (x @ up_w.to(F32).T)) @ down_w.to(F32).T


def clamped_swiglu(g: torch.Tensor, u: torch.Tensor, limit: float) -> torch.Tensor:
    """The routed experts' activation: gate clamped above only, up clamped on both sides."""
    return F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)


def route(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_idx [S, k] long, topk_w [S, k] fp32)`` of HF ``HYV4MoE``'s router (``reference/hy4/model.py _moe``).

    The picks are returned in descending order of the biased choice score; HF uses ``sorted=False``, which only
    permutes the k slots, so compare picks as sets."""
    scores = torch.sigmoid(x.to(F32) @ gate_w.to(F32).T)
    choice = scores + bias.to(F32)
    E = scores.shape[-1]
    if n_group > 1:
        grp = choice.view(-1, n_group, E // n_group).topk(2, dim=-1)[0].sum(-1)
        keep = torch.zeros_like(grp).scatter_(1, grp.topk(topk_group, dim=-1)[1], 1.0)
        choice = choice.masked_fill(~keep.repeat_interleave(E // n_group, dim=1).bool(), float("-inf"))
    top_i = choice.topk(top_k, dim=-1, sorted=True)[1]
    top_w = scores.gather(1, top_i)
    if norm_topk_prob:
        top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
    return top_i, top_w * routed_scaling_factor


def _expert_groups(top_i: torch.Tensor, num_experts: int):
    """Yield ``(expert, token rows, top-k slots)`` for every expert some token picked."""
    k = top_i.shape[-1]
    flat = top_i.flatten()
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=num_experts).tolist()
    start = 0
    for e, c in enumerate(counts):
        if c:
            sel = order[start : start + c]
            start += c
            yield e, sel // k, sel % k


def routed_moe(
    x: torch.Tensor,
    top_i: torch.Tensor,
    top_w: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    limit: float,
) -> torch.Tensor:
    """HF ``HYV4MoE``'s routed half for given picks: ``sum_k w_k * expert_k(x)``, clamped SwiGLU per expert.

    ``gate_up`` ``[E, 2I, H]`` (gate rows then up rows), ``down`` ``[E, H, I]``, as stored in the checkpoint."""
    x = x.to(F32)
    out = torch.zeros_like(x)
    for e, tok, slot in _expert_groups(top_i, gate_up.shape[0]):
        g, u = (x[tok] @ gate_up[e].to(F32).T).chunk(2, dim=-1)
        y = (clamped_swiglu(g, u, limit) @ down[e].to(F32).T) * top_w[tok, slot].unsqueeze(-1)
        out.index_add_(0, tok, y)
    return out


def clamp_fractions(
    x: torch.Tensor, top_i: torch.Tensor, gate_up: torch.Tensor, *, limit: float
) -> tuple[float, float]:
    """Fractions of routed gate values above ``limit`` and of routed up values outside ``[-limit, limit]``."""
    x = x.to(F32)
    n = g_hit = u_hit = 0
    for e, tok, _ in _expert_groups(top_i, gate_up.shape[0]):
        g, u = (x[tok] @ gate_up[e].to(F32).T).chunk(2, dim=-1)
        g_hit += int((g > limit).sum())
        u_hit += int((u.abs() > limit).sum())
        n += g.numel()
    return (g_hit / n, u_hit / n) if n else (0.0, 0.0)


def moe(
    x: torch.Tensor,
    *,
    gate_w: torch.Tensor,
    bias: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    shared: dict,
    top_k: int,
    limit: float,
    routed_scaling_factor: float,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
) -> dict:
    """The whole HF ``HYV4MoE`` block (routed + shared): ``{"topk_idx", "topk_w", "routed", "shared", "out"}``.
    ``shared`` holds the shared expert's ``gate_proj`` / ``up_proj`` / ``down_proj`` (HF layout)."""
    top_i, top_w = route(
        x,
        gate_w,
        bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )
    routed = routed_moe(x, top_i, top_w, gate_up, down, limit=limit)
    sh = swiglu_mlp(x, shared["gate_proj"], shared["up_proj"], shared["down_proj"])
    return {"topk_idx": top_i, "topk_w": top_w, "routed": routed, "shared": sh, "out": routed + sh}
