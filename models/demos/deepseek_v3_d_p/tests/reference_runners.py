# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Reference runners for each model variant.

These run the variant's reference on CPU and return a torch tensor
suitable for PCC comparison against the TT output. Variants without a
bundled reference return `None` and the comparison is skipped at the call
site.
"""

from copy import copy, deepcopy
from typing import Optional

import torch
from transformers.activations import ACT2FN

from models.demos.common.prefill.adapter import PrefillModelAdapter as TestVariant


def _build_act_fn(cfg):
    """The activation an upstream MLP would build for this config.

    Mirrors KimiMLP/KimiBlockSparseMLP rather than going straight to ACT2FN, which has no "situ"
    entry -- so the override below can name either activation for either half.
    """
    if getattr(cfg, "hidden_act", None) == "situ":
        from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import (
            SituAndMul,
            _get_situ_activation_params,
        )

        beta, linear_beta = _get_situ_activation_params(cfg)
        return SituAndMul(beta=beta, linear_beta=linear_beta)
    return ACT2FN[cfg.hidden_act]


def run_reference_moe(
    variant: TestVariant,
    *,
    config,
    gate_weights,
    routed_expert_weights,
    shared_expert_weights,
    x,
    latent_weights=None,
    hidden_act: Optional[str] = None,
    shared_hidden_act: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MoE reference on CPU.

    ``hidden_act`` overrides the config's GLU activation; ``shared_hidden_act`` overrides it for the
    shared expert alone. The upstream block picks one activation for routed and shared alike, but a
    device may implement the two halves differently (Kimi-K3: SiTU-GLU in the fused routed-expert
    kernel, SiLU for the shared expert), and this reference is only a fair cross-check if it splits
    the same way.
    """
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
    if hasattr(cfg, "num_experts"):
        cfg.num_experts = cfg.n_routed_experts
    if routed_expert_weights:
        cfg.moe_intermediate_size = routed_expert_weights[0]["gate_proj"].shape[0]
        # Kimi-K3's routed experts read the latent width, not hidden_size.
        if getattr(cfg, "routed_expert_hidden_size", None) is not None:
            cfg.routed_expert_hidden_size = routed_expert_weights[0]["gate_proj"].shape[1]
    if hidden_act is not None:
        cfg.hidden_act = hidden_act
    moe = variant.reference_moe_cls(cfg)
    # The shared expert reads hidden_act off the config object it was handed, so giving it a copy
    # with a different activation retargets that half only.
    if shared_hidden_act is not None and shared_hidden_act != cfg.hidden_act:
        shared = getattr(moe, "shared_experts", None)
        assert shared is not None, "shared_hidden_act was given but the reference has no shared expert"
        shared_cfg = copy(cfg)
        shared_cfg.hidden_act = shared_hidden_act
        shared.config = shared_cfg
        shared.act_fn = _build_act_fn(shared_cfg)
    moe.load_state_dict(
        _pack_reference_moe_state_dict(moe, gate_weights, routed_expert_weights, shared_expert_weights, latent_weights),
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
    # load_state_dict is non-strict (references carry submodules the TT weight dict doesn't), so a
    # missing weight silently stays at its random init and only shows up as a baffling PCC miss.
    # Gate-bearing variants (Kimi-K3) are checked explicitly for that reason.
    if getattr(config, "mla_use_output_gate", False):
        assert "g_proj.weight" in weights, (
            "config declares mla_use_output_gate but the MLA weight dict has no 'g_proj.weight'; "
            "the reference would run with a randomly-initialised gate"
        )
    attn.load_state_dict(weights, strict=False)
    attn = attn.eval().to(torch.bfloat16)
    causal = torch.triu(torch.full((q_len, q_len), float("-inf"), dtype=hidden_states.dtype), diagonal=1)
    with torch.no_grad():
        out = attn(
            hidden_states=hidden_states,
            attention_mask=causal[None, None],
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
        )
    # DeepseekV3Attention returns (out, attn_weights, past_kv); Kimi-K3's KimiMLAAttention returns a
    # bare tensor (upstream shape). Accept either.
    return out[0] if isinstance(out, tuple) else out


def _pack_reference_moe_state_dict(moe, gate_weights, routed_expert_weights, shared_expert_weights, latent_weights):
    """Pack TT-side weight dicts into ``moe``'s own key layout, for a strict load.

    Read off the constructed module rather than a per-variant table: Kimi-K3 names its routed
    projections w1/w3/w2 and adds the latent pair plus its norm, and anything else it grows would
    surface as a strict-load error naming the key.
    """
    sd = {
        "gate.weight": gate_weights["weight"],
        "gate.e_score_correction_bias": gate_weights["e_score_correction_bias"],
        "shared_experts.gate_proj.weight": shared_expert_weights["gate_proj"],
        "shared_experts.up_proj.weight": shared_expert_weights["up_proj"],
        "shared_experts.down_proj.weight": shared_expert_weights["down_proj"],
    }
    src = ("gate_proj", "up_proj", "down_proj")
    dst = ("w1", "w3", "w2") if hasattr(moe.experts[0], "w1") else src
    for i, w in enumerate(routed_expert_weights):
        for proj, name in zip(src, dst):
            sd[f"experts.{i}.{name}.weight"] = w[proj]
    if hasattr(moe, "routed_expert_down_proj"):
        sd["routed_expert_down_proj.weight"] = latent_weights["down_proj"]
        sd["routed_expert_up_proj.weight"] = latent_weights["up_proj"]
        if hasattr(moe, "routed_expert_norm"):
            sd["routed_expert_norm.weight"] = latent_weights["norm"]
    return {k: v.to(torch.bfloat16) for k, v in sd.items()}
