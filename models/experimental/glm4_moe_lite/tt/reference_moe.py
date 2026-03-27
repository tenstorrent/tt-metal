# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.experimental.glm4_moe_lite.tt.weights import LazyStateDict, load_glm_lazy_state_dict


@dataclass(frozen=True)
class MoERefOutputs:
    # Routing
    router_logits: torch.Tensor  # [T, E] float32 (pre-sigmoid)
    topk_indices: torch.Tensor  # [T, K] int64
    topk_weights: torch.Tensor  # [T, K] float32 (post-renorm + scaling)

    # Compute
    shared_out: torch.Tensor  # [T, H] float32
    routed_out: torch.Tensor  # [T, H] float32
    moe_out: torch.Tensor  # [T, H] float32 (shared + routed)


def _load_hparams_from_snapshot(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _route_tokens_to_experts_reference(
    router_logits: torch.Tensor,
    *,
    e_score_correction_bias: torch.Tensor,  # [E] float32
    n_group: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation matching HF `Glm4MoeLiteMoE.route_tokens_to_experts`.

    Returns:
    - topk_indices: [T, K] int64
    - topk_weights: [T, K] float32 (post-renorm + routed scaling)
    """
    if router_logits.ndim != 2:
        raise ValueError(f"expected router_logits [T,E], got shape={tuple(router_logits.shape)}")

    scores = router_logits.sigmoid()  # [T,E]
    scores_for_choice = scores + e_score_correction_bias.view(1, -1)

    if n_group <= 0:
        raise ValueError(f"n_group must be > 0, got n_group={n_group}")
    if topk_group <= 0 or topk_group > n_group:
        raise ValueError(f"topk_group must be in [1,n_group], got topk_group={topk_group} n_group={n_group}")

    num_tokens, num_experts = scores_for_choice.shape
    if num_experts % n_group != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by n_group={n_group}")
    experts_per_group = num_experts // n_group

    # group_scores: [T, n_group]
    group_scores = scores_for_choice.view(num_tokens, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [T, topk_group]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(num_tokens, num_experts)
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]  # [T,K]
    topk_weights = scores.gather(1, topk_indices)  # [T,K]
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * float(routed_scaling_factor)
    return topk_indices, topk_weights


def _dense_mlp_reference(
    x: torch.Tensor,  # [T,H] float32
    *,
    w_gate: torch.Tensor,  # [I,H] bf16/fp32
    w_up: torch.Tensor,  # [I,H]
    w_down: torch.Tensor,  # [H,I]
) -> torch.Tensor:
    gate = torch.nn.functional.linear(x, w_gate.to(dtype=torch.float32))
    up = torch.nn.functional.linear(x, w_up.to(dtype=torch.float32))
    x_ff = torch.nn.functional.silu(gate) * up
    return torch.nn.functional.linear(x_ff, w_down.to(dtype=torch.float32))


@torch.no_grad()
def run_layer_moe_reference_from_hidden_states(
    snapshot_dir: Path,
    *,
    layer_idx: int,
    hidden_states: torch.Tensor,
) -> MoERefOutputs:
    """
    CPU reference for a single GLM4 MoE layer's MLP block (shared + routed experts).

    Notes:
    - This matches HF routing semantics (sigmoid + correction bias for choice, gather
      from unbiased sigmoid scores, optional renorm, routed scaling factor).
    - This does NOT include the decoder-layer residual add; it produces the MLP
      block output to be added to the residual path by the caller.
    """
    snapshot_dir = Path(snapshot_dir)
    hparams = _load_hparams_from_snapshot(snapshot_dir)

    if hidden_states.ndim == 4:
        # Accept TT-style [1,1,T,H] and flatten.
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    if hidden_states.ndim != 2:
        raise ValueError(f"expected hidden_states [T,H] (or [1,1,T,H]), got shape={tuple(hidden_states.shape)}")

    hidden_size = int(hparams.hidden_size)
    if int(hidden_states.shape[1]) != hidden_size:
        raise ValueError(f"hidden_states last dim must be {hidden_size}, got {int(hidden_states.shape[1])}")

    # Match HF behavior: router logits computed in fp32 from fp32(hidden_states).
    x = hidden_states.to(dtype=torch.float32)

    state: LazyStateDict = load_glm_lazy_state_dict(snapshot_dir, num_layers=int(hparams.num_hidden_layers))

    # Gate/router weights (routed experts only).
    w_gate = state[f"model.layers.{int(layer_idx)}.mlp.gate.weight"]  # [E,H] bf16
    e_bias = state[f"model.layers.{int(layer_idx)}.mlp.gate.e_score_correction_bias"]  # [E] fp32

    router_logits = torch.nn.functional.linear(x, w_gate.to(dtype=torch.float32))  # [T,E]
    topk_indices, topk_weights = _route_tokens_to_experts_reference(
        router_logits,
        e_score_correction_bias=e_bias.to(dtype=torch.float32),
        n_group=int(getattr(hparams, "n_group", 1)),
        topk_group=int(getattr(hparams, "topk_group", 1)),
        top_k=int(hparams.num_experts_per_tok),
        norm_topk_prob=bool(getattr(hparams, "norm_topk_prob", True)),
        routed_scaling_factor=float(getattr(hparams, "routed_scaling_factor", 1.0)),
    )

    # Shared expert (dense) weights.
    w_shared_gate = state[f"model.layers.{int(layer_idx)}.mlp.shared_experts.gate_proj.weight"]
    w_shared_up = state[f"model.layers.{int(layer_idx)}.mlp.shared_experts.up_proj.weight"]
    w_shared_down = state[f"model.layers.{int(layer_idx)}.mlp.shared_experts.down_proj.weight"]
    shared_out = _dense_mlp_reference(x, w_gate=w_shared_gate, w_up=w_shared_up, w_down=w_shared_down)  # [T,H]

    # Routed experts: weighted sum across top-k experts.
    num_tokens, _ = topk_indices.shape
    routed_out = torch.zeros((num_tokens, hidden_size), dtype=torch.float32)

    unique_experts = torch.unique(topk_indices)
    for expert_id in unique_experts.tolist():
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= int(hparams.n_routed_experts):
            raise ValueError(f"expert id out of range: {expert_id}")

        # Find (token_idx, k_pos) where this expert is selected.
        mask = topk_indices == expert_id  # [T,K]
        if not mask.any():
            continue
        token_idx, k_pos = torch.where(mask)
        x_sel = x[token_idx]  # [N,H]

        w1 = state[f"model.layers.{int(layer_idx)}.mlp.experts.{expert_id}.gate_proj.weight"]
        w3 = state[f"model.layers.{int(layer_idx)}.mlp.experts.{expert_id}.up_proj.weight"]
        w2 = state[f"model.layers.{int(layer_idx)}.mlp.experts.{expert_id}.down_proj.weight"]

        out = _dense_mlp_reference(x_sel, w_gate=w1, w_up=w3, w_down=w2)  # [N,H]
        w = topk_weights[token_idx, k_pos].to(dtype=torch.float32).unsqueeze(-1)  # [N,1]
        routed_out.index_add_(0, token_idx, out * w)

    moe_out = shared_out + routed_out

    return MoERefOutputs(
        router_logits=router_logits.cpu(),
        topk_indices=topk_indices.cpu(),
        topk_weights=topk_weights.cpu(),
        shared_out=shared_out.cpu(),
        routed_out=routed_out.cpu(),
        moe_out=moe_out.cpu(),
    )


__all__ = ["MoERefOutputs", "run_layer_moe_reference_from_hidden_states"]
