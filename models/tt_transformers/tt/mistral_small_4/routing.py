# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mistral4 MoE routing: grouped top-k selection (``route_tokens_to_experts``)."""

from __future__ import annotations

import torch

import ttnn


def route_tokens_from_probs_torch(
    router_probs: torch.Tensor,
    *,
    n_group: int,
    n_routed_experts: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HF ``Mistral4MoE.route_tokens_to_experts`` logic **after** the softmax on router logits.

    ``router_probs``: ``[N, n_routed_experts]``, rows are probability vectors (sum 1).
    """
    rl = router_probs
    group_scores = rl.view(-1, n_group, n_routed_experts // n_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, n_routed_experts // n_group).reshape(-1, n_routed_experts)
    scores_for_choice = rl.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = rl.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


def route_tokens_to_experts_reference_torch(
    router_logits: torch.Tensor,
    *,
    n_group: int,
    n_routed_experts: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full HF path: softmax on logits, then :func:`route_tokens_from_probs_torch`."""
    router_probs = router_logits.softmax(-1)
    return route_tokens_from_probs_torch(
        router_probs,
        n_group=n_group,
        n_routed_experts=n_routed_experts,
        topk_group=topk_group,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )


def router_softmax_then_route_bf16(
    mesh_device,
    router_logits_ne: torch.Tensor,
    *,
    n_group: int,
    n_routed_experts: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Softmax router logits on device (bf16), then run grouped top-k routing on **host** (torch).

    This validates device softmax against the same downstream math as HF; weights/indices use
    probabilities read back from the device in bf16.

    Args:
        router_logits_ne: pre-softmax logits ``[N, n_routed_experts]``.
    """
    if router_logits_ne.ndim != 2:
        raise ValueError(f"expected router logits [N, E], got {tuple(router_logits_ne.shape)}")
    n, e = int(router_logits_ne.shape[0]), int(router_logits_ne.shape[1])
    if e != n_routed_experts:
        raise ValueError(f"last dim {e} != n_routed_experts {n_routed_experts}")
    if n_routed_experts % n_group != 0:
        raise ValueError("n_routed_experts must be divisible by n_group")
    if (n_routed_experts // n_group) < 2:
        raise ValueError("experts per group must be >= 2 (HF uses topk(..., k=2) within each group)")

    x_bf16 = router_logits_ne.to(torch.bfloat16)
    tt = ttnn.from_torch(
        x_bf16.reshape(1, 1, n, e),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_p = ttnn.softmax(tt, dim=-1)
    ttnn.deallocate(tt)
    probs = ttnn.to_torch(tt_p, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(tt_p)
    while probs.ndim > 2:
        probs = probs.squeeze(0)
    probs = probs.reshape(n, e).to(torch.bfloat16)

    return route_tokens_from_probs_torch(
        probs,
        n_group=n_group,
        n_routed_experts=n_routed_experts,
        topk_group=topk_group,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )


def router_softmax_then_route_device_bf16(
    mesh_device,
    router_logits_ne: torch.Tensor,
    *,
    n_group: int,
    n_routed_experts: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full grouped routing on device (bf16), returning host indices/weights.

    This is the same algorithm as :func:`route_tokens_to_experts_reference_torch`, but implemented with
    ``ttnn`` ops:

    - softmax over experts
    - per-group score = sum(top-2 experts within the group)
    - select top-``topk_group`` groups
    - mask to active groups and take top-``top_k`` experts
    - optionally renormalize selected probs and apply scaling

    Returns host tensors because downstream HF expert execution uses torch indices/weights.
    """
    if router_logits_ne.ndim != 2:
        raise ValueError(f"expected router logits [N, E], got {tuple(router_logits_ne.shape)}")
    n, e = int(router_logits_ne.shape[0]), int(router_logits_ne.shape[1])
    if e != int(n_routed_experts):
        raise ValueError(f"last dim {e} != n_routed_experts {n_routed_experts}")
    if int(n_routed_experts) % int(n_group) != 0:
        raise ValueError("n_routed_experts must be divisible by n_group")
    experts_per_group = int(n_routed_experts) // int(n_group)
    if experts_per_group < 2:
        raise ValueError("experts per group must be >= 2 (HF uses topk(..., k=2) within each group)")
    if top_k <= 0 or topk_group <= 0:
        raise ValueError("top_k and topk_group must be positive")

    TOPK_MIN_WIDTH = 64

    # [N, E] -> [1, 1, N, E] on device
    x_bf16 = router_logits_ne.to(torch.bfloat16)
    logits_tt = ttnn.from_torch(
        x_bf16.reshape(1, 1, n, e),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    probs_tt = ttnn.softmax(logits_tt, dim=-1)
    ttnn.deallocate(logits_tt)

    # Grouped scores: reshape to [1,1,N*n_group, experts_per_group] then top-2 per group.
    grouped = ttnn.reshape(probs_tt, (1, 1, n * int(n_group), experts_per_group))
    if grouped.shape[3] < TOPK_MIN_WIDTH:
        grouped = ttnn.pad(
            grouped,
            [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - grouped.shape[3])],
            value=-float("inf"),
        )
    top2_vals, _top2_idx = ttnn.topk(grouped, k=2, dim=-1, largest=True, sorted=False)
    ttnn.deallocate(grouped)
    ttnn.deallocate(_top2_idx)
    group_scores = ttnn.sum(top2_vals, dim=-1, keepdim=True)  # [1,1,N*n_group,1]
    ttnn.deallocate(top2_vals)
    group_scores = ttnn.reshape(group_scores, (1, 1, n, int(n_group)))  # [1,1,N,G]

    # Select active groups.
    if group_scores.shape[3] < TOPK_MIN_WIDTH:
        group_scores = ttnn.pad(
            group_scores,
            [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - group_scores.shape[3])],
            value=-float("inf"),
        )
    _gv, top_group_idx = ttnn.topk(group_scores, k=int(topk_group), dim=-1, largest=True, sorted=False)
    ttnn.deallocate(group_scores)
    ttnn.deallocate(_gv)

    # Build group mask [1,1,N,G] (row-major for scatter), then expand to experts.
    top_group_idx = ttnn.to_layout(top_group_idx, ttnn.ROW_MAJOR_LAYOUT)
    group_mask = ttnn.zeros((1, 1, n, int(n_group)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device)
    group_src = ttnn.ones((1, 1, n, int(topk_group)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device)
    group_mask = ttnn.scatter(input=group_mask, dim=-1, index=top_group_idx, src=group_src)
    ttnn.deallocate(top_group_idx)
    ttnn.deallocate(group_src)

    group_mask = ttnn.repeat_interleave(group_mask, experts_per_group, dim=-1)  # -> [1,1,N,E]
    group_mask = ttnn.to_layout(group_mask, ttnn.TILE_LAYOUT)

    # Mask out inactive experts by adding ~(-1e9) to their probs so topk won't pick them.
    neg_inf = ttnn.multiply(ttnn.subtract(group_mask, 1.0), 1e9)
    ttnn.deallocate(group_mask)
    masked = ttnn.add(probs_tt, neg_inf)
    ttnn.deallocate(neg_inf)

    # Top-k experts across all experts.
    topk_masked_vals, topk_idx = ttnn.topk(masked, k=int(top_k), dim=-1, largest=True, sorted=False)
    ttnn.deallocate(masked)
    ttnn.deallocate(topk_masked_vals)

    # Gather true probs (unmasked) at chosen indices.
    topk_probs = ttnn.gather(probs_tt, dim=-1, index=topk_idx)
    ttnn.deallocate(probs_tt)

    if norm_topk_prob:
        denom = ttnn.sum(topk_probs, dim=-1, keepdim=True) + 1e-20
        topk_probs = ttnn.div(topk_probs, denom)
        ttnn.deallocate(denom)

    if float(routed_scaling_factor) != 1.0:
        topk_probs = ttnn.mul(topk_probs, float(routed_scaling_factor))

    # Return host tensors: [N, top_k]
    idx_host = ttnn.to_torch(topk_idx, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    w_host = ttnn.to_torch(topk_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(topk_idx)
    ttnn.deallocate(topk_probs)

    while idx_host.ndim > 2:
        idx_host = idx_host.squeeze(0)
    while w_host.ndim > 2:
        w_host = w_host.squeeze(0)

    idx_host = idx_host.reshape(n, int(top_k)).to(torch.long)
    w_host = w_host.reshape(n, int(top_k)).to(torch.bfloat16)
    return idx_host, w_host
