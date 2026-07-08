# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.1 MoE FFN CPU reference (plain routed MoE, no device dispatch/combine).

The routing is the *same* noaux_tc gate the device uses (and that test_ttnn_moe validates the on-device
gate against): reference.modeling_deepseek.MoEGate with scoring_func="sigmoid", topk_method="noaux_tc",
norm_topk_prob=True, routed_scaling_factor=route_scale. Each token's selected experts are applied with
TorchExpert (silu-gated FFN) and summed with the shared expert. This is the mathematical MoE result
that the distributed TtMoe (dispatch -> experts -> combine) computes, so it composes into the block
reference without replicating the dispatch machinery.
"""

from types import SimpleNamespace

import torch

from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert


def glm_moe_reference(
    hidden_states: torch.Tensor,
    *,
    gate_weights: dict,
    routed_expert_weights: list[dict],
    shared_expert_weights: dict,
    emb_dim: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Plain routed-MoE forward. hidden_states [1, seq, hidden] -> [1, seq, hidden] (bf16)."""
    b, s, h = hidden_states.shape
    flat = hidden_states.reshape(-1, h)
    n_routed = len(routed_expert_weights)

    gate = MoEGate(
        SimpleNamespace(
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=True,
            hidden_size=emb_dim,
        ),
        use_bitonic_sort=False,
    )
    gate.weight.data = gate_weights["weight"]
    # Match the device: TtMoEGatePrefill stores e_score_correction_bias as bf16 (even in DEVICE_FP32,
    # which only makes the *logits* fp32). GLM's real bias is ~34.5 with a ~±0.14 inter-expert spread —
    # finer than bf16's ~0.125-0.25 resolution near 34.5 — and it dominates top-k selection, so an fp32
    # reference bias would pick different experts than the device (bf16). Quantize to bf16 here so the
    # reference routes like the device. (The bf16-bias precision itself is a separate model-accuracy
    # concern, not what this block test validates.)
    gate.e_score_correction_bias.data = gate_weights["e_score_correction_bias"].to(torch.bfloat16).float()
    # MoEGate.forward expects [bsz, seq, hidden] (it unpacks 3 dims); it flattens internally and
    # returns topk_idx / topk_weight as [tokens, top_k].
    topk_idx, topk_weight = gate(hidden_states.float())

    routed_hidden = routed_expert_weights[0]["gate_proj"].shape[0]
    out = torch.zeros_like(flat, dtype=torch.float32)
    for e in range(n_routed):
        sel = topk_idx == e  # [tokens, top_k]
        if not bool(sel.any()):
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        expert = TorchExpert(emb_dim, routed_hidden, torch_weights=routed_expert_weights[e]).eval().to(torch.bfloat16)
        with torch.no_grad():
            ex_out = expert(flat[tok])
        out[tok] += topk_weight[tok, slot].unsqueeze(-1).float() * ex_out.float()

    shared_hidden = shared_expert_weights["gate_proj"].shape[0]
    shared = TorchExpert(emb_dim, shared_hidden, torch_weights=shared_expert_weights).eval().to(torch.bfloat16)
    with torch.no_grad():
        out += shared(flat).float()

    return out.reshape(b, s, h).to(hidden_states.dtype)
