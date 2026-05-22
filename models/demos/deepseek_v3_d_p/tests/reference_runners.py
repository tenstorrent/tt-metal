# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Upstream reference runners — per-variant HF cross-check helpers.

These run the variant's bundled upstream reference (e.g., HF
`DeepseekV3MoE` / `DeepseekV3Attention`) on CPU and return a torch tensor
suitable for PCC comparison against the TT output. Variants without a
bundled reference return `None` and the comparison is skipped at the call
site.

Kept separate from `model_variants.py` so the variant taxonomy stays a pure
registry; this file is where reference-runner behavior lives.
"""

from typing import Optional

import torch

from models.demos.deepseek_v3_d_p.tests.model_variants import ModelVariant


def run_reference_moe(
    variant: ModelVariant,
    *,
    gate_weights,
    routed_expert_weights,
    shared_expert_weights,
    x,
    num_routed_experts: Optional[int] = None,
    num_experts_per_tok: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MoE reference on CPU. None if not bundled.

    `num_routed_experts` / `num_experts_per_tok` override the variant's canonical
    values — required when the test runs a scaled-down expert count (the
    reference model and the supplied state-dict must agree on the expert count
    or `load_state_dict(strict=True)` raises).
    """
    if variant.reference_moe_cls is None:
        return None
    config = variant.build_reference_config()
    if num_routed_experts is not None:
        config.n_routed_experts = num_routed_experts
    if num_experts_per_tok is not None:
        config.num_experts_per_tok = num_experts_per_tok
    moe = variant.reference_moe_cls(config)
    moe.load_state_dict(
        _pack_reference_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights),
        strict=True,
    )
    moe = moe.eval().to(torch.bfloat16)
    with torch.no_grad():
        return moe(x.to(torch.bfloat16))


def run_reference_mla(
    variant: ModelVariant,
    *,
    weights,
    hidden_states,
    position_ids,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MLA reference on CPU. None if not bundled.

    Caller owns the seq_len budget: HF DeepseekV3Attention materializes
    `[bsz, heads, q_len, q_len]` in fp32 — gate the call yourself at the test
    site when running with a large q_len.
    """
    if variant.reference_attention_cls is None:
        return None
    _, q_len, _ = hidden_states.shape
    attn = variant.reference_attention_cls(variant.build_reference_config(), layer_idx=0)
    attn.load_state_dict(weights, strict=False)
    attn = attn.eval().to(torch.bfloat16)
    causal = torch.triu(torch.full((q_len, q_len), float("-inf"), dtype=hidden_states.dtype), diagonal=1)
    with torch.no_grad():
        out, _, _ = attn(
            hidden_states=hidden_states,
            attention_mask=causal[None, None],
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
        )
    return out


def _pack_reference_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights) -> dict:
    sd = {
        "gate.weight": gate_weights["weight"],
        "gate.e_score_correction_bias": gate_weights["e_score_correction_bias"],
        "shared_experts.gate_proj.weight": shared_expert_weights["gate_proj"],
        "shared_experts.up_proj.weight": shared_expert_weights["up_proj"],
        "shared_experts.down_proj.weight": shared_expert_weights["down_proj"],
    }
    for i, w in enumerate(routed_expert_weights):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            sd[f"experts.{i}.{proj}.weight"] = w[proj]
    return {k: v.to(torch.bfloat16) for k, v in sd.items()}
