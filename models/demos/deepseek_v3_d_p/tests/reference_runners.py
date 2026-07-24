# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Reference runners for each model variant.

These run the variant's reference on CPU and return a torch tensor
suitable for PCC comparison against the TT output. Variants without a
bundled reference return `None` and the comparison is skipped at the call
site.
"""

from copy import deepcopy
from typing import Optional

import torch

from models.demos.common.prefill.adapter import PrefillModelAdapter as TestVariant


def run_reference_moe(
    variant: TestVariant,
    *,
    config,
    gate_weights,
    routed_expert_weights,
    shared_expert_weights,
    x,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MoE reference on CPU."""
    if variant.reference_moe_cls is None:
        return None
    # Test params can override the variant's default MoE dims (expert count, hidden/intermediate size —
    # e.g. GLM-5.1's 256 experts / 6144 hidden vs the deepseek_v3 variant's 7168), so patch the reference
    # config from the actual generated weight shapes: gate.weight is [n_experts, hidden], each expert's
    # gate_proj is [moe_intermediate, hidden]. Without this the reference is built at the variant default
    # and load_state_dict fails with a size mismatch.
    cfg = deepcopy(config)
    cfg.n_routed_experts = gate_weights["weight"].shape[0]
    cfg.hidden_size = gate_weights["weight"].shape[1]
    if routed_expert_weights:
        cfg.moe_intermediate_size = routed_expert_weights[0]["gate_proj"].shape[0]
    moe = variant.reference_moe_cls(cfg)
    moe.load_state_dict(
        _pack_reference_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights),
        strict=True,
    )
    moe = moe.eval().to(torch.bfloat16)
    with torch.no_grad():
        return moe(x.to(torch.bfloat16))


def run_reference_mla(
    variant: TestVariant,
    *,
    config,
    weights,
    hidden_states,
    position_ids,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MLA reference on CPU."""
    if variant.reference_attention_cls is None:
        return None
    _, q_len, _ = hidden_states.shape
    attn = variant.reference_attention_cls(config, layer_idx=0)
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
