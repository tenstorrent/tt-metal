# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-variant test configurations for the DeepSeek V3 architecture family.

DSv3 and Kimi K2.6 share the DeepSeek V3 architecture (Kimi ships with
`architectures: ["DeepseekV3ForCausalLM"]`); only hyperparameters differ.
This module bundles those differences so that a single parametrized test
body covers both variants — see test_ttnn_moe.py and test_mla.py.

Public surface:
- ModelVariant, GateRouting              — variant dataclasses
- apply_gate_overrides_moe(variant)      — runtime gate-config patch on TT + Torch MoE
- run_reference_moe(variant, …)          — variant's upstream MoE reference cross-check
- run_reference_mla(variant, …)          — variant's upstream MLA reference cross-check
- MODEL_VARIANTS                         — {name: ModelVariant}

Adding a new variant: define a `_build_<name>_reference_config()` and a
`<NAME> = ModelVariant(...)` instance, then append to the list at the bottom.
"""

from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional, Tuple

import torch

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config as DSv3RefConfig
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention as DSv3RefAttention
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE as DSv3RefMoE
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.experimental.kimi_k26.reference.configuration_deepseek import DeepseekV3Config as KimiRefConfig
from models.experimental.kimi_k26.reference.modeling_deepseek import DeepseekV3Attention as KimiRefAttention
from models.experimental.kimi_k26.reference.modeling_deepseek import DeepseekV3MoE as KimiRefMoE


@dataclass(frozen=True)
class GateRouting:
    """MoE gate routing knobs (DSv3 defaults; Kimi overrides to 1/1/2.827)."""

    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5


@dataclass(frozen=True)
class ModelVariant:
    """One variant of the DeepSeek V3 architecture family.

    Mandatory fields (`name`, `gate`, `build_reference_config`) drive the
    existing DSv3 reference path (TorchMoe / create_mla_reference) with this
    variant's hyperparameters.

    Optional `reference_*_cls` enable an upstream cross-check that runs in
    addition to the always-on DSv3 reference comparison.

    Test-environment constraints (`supported_meshes`, `required_gate_fallback_mode`,
    `supports_pretrained`) let tests gate cases generically — no variant-name
    conditionals required in the test body.
    """

    name: str
    gate: GateRouting
    build_reference_config: Callable[[], "PretrainedConfig"]

    # Optional capabilities — None ⇒ skip that cross-check.
    reference_moe_cls: Optional[type] = None
    reference_attention_cls: Optional[type] = None

    moe_pcc_threshold: float = 0.96
    mla_pcc_threshold: float = 0.98

    # Test-env constraints — None ⇒ no constraint.
    supported_meshes: Optional[FrozenSet[Tuple[int, int]]] = None
    required_gate_fallback_mode: Optional[GateComputeMode] = None
    supports_pretrained: bool = True


def apply_gate_overrides_moe(tt_moe, torch_moe, variant: ModelVariant) -> None:
    """Patch gate routing on TT + Torch MoE. DSv3 values are a no-op patch.

    Either side may be None.
    """
    g = variant.gate
    if tt_moe is not None:
        cfg = tt_moe.gate.config
        cfg.n_expert_groups, cfg.n_limited_groups, cfg.route_scale = (
            g.n_expert_groups,
            g.n_limited_groups,
            g.route_scale,
        )
    if torch_moe is not None:
        gate = torch_moe.gate
        gate.n_group, gate.topk_group, gate.routed_scaling_factor = (
            g.n_expert_groups,
            g.n_limited_groups,
            g.route_scale,
        )


def run_reference_moe(
    variant: ModelVariant,
    *,
    gate_weights,
    routed_expert_weights,
    shared_expert_weights,
    x,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MoE reference on CPU. None if not bundled."""
    if variant.reference_moe_cls is None:
        return None
    moe = variant.reference_moe_cls(variant.build_reference_config())
    moe.load_state_dict(
        _pack_reference_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights),
        strict=False,
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
    """Forward the variant's upstream MLA reference on CPU. None if not bundled."""
    if variant.reference_attention_cls is None:
        return None
    attn = variant.reference_attention_cls(variant.build_reference_config(), layer_idx=0)
    attn.load_state_dict(weights, strict=False)
    attn = attn.eval().to(torch.bfloat16)
    _, q_len, _ = hidden_states.shape
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
    """Repack random TT-style weights into the reference DeepseekV3MoE state-dict layout."""
    sd = {
        "gate.weight": gate_weights["weight"].to(torch.bfloat16),
        "gate.e_score_correction_bias": gate_weights["e_score_correction_bias"].to(torch.bfloat16),
        "shared_experts.gate_proj.weight": shared_expert_weights["gate_proj"].to(torch.bfloat16),
        "shared_experts.up_proj.weight": shared_expert_weights["up_proj"].to(torch.bfloat16),
        "shared_experts.down_proj.weight": shared_expert_weights["down_proj"].to(torch.bfloat16),
    }
    for i, w in enumerate(routed_expert_weights):
        sd[f"experts.{i}.gate_proj.weight"] = w["gate_proj"].to(torch.bfloat16)
        sd[f"experts.{i}.up_proj.weight"] = w["up_proj"].to(torch.bfloat16)
        sd[f"experts.{i}.down_proj.weight"] = w["down_proj"].to(torch.bfloat16)
    return sd


# ---------------------------------------------------------------------------
# Per-variant config builders. One per variant; produces the full reference
# config (MoE fields are unused by MLA paths and vice versa, no harm).
# ---------------------------------------------------------------------------


def _build_dsv3_reference_config() -> DSv3RefConfig:
    c = DeepSeekV3Config
    return DSv3RefConfig(
        vocab_size=c.VOCAB_SIZE,
        hidden_size=c.EMB_SIZE,
        intermediate_size=c.INTERMEDIATE_SIZE,
        moe_intermediate_size=c.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=c.NUM_LAYERS,
        num_attention_heads=c.NUM_ATTENTION_HEADS,
        num_key_value_heads=c.NUM_ATTENTION_HEADS,
        q_lora_rank=c.Q_LORA_RANK,
        kv_lora_rank=c.KV_LORA_RANK,
        qk_nope_head_dim=c.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=c.QK_ROPE_HEAD_DIM,
        v_head_dim=c.V_HEAD_DIM,
        rms_norm_eps=c.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=c.NUM_DENSE_LAYERS,
        n_routed_experts=c.NUM_ROUTED_EXPERTS,
        n_shared_experts=c.NUM_SHARED_EXPERTS,
        num_experts_per_tok=c.NUM_EXPERTS_PER_TOKEN,
        n_group=c.NUM_EXPERT_GROUPS,
        topk_group=c.NUM_LIMITED_GROUPS,
        routed_scaling_factor=c.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


def _build_kimi_reference_config() -> KimiRefConfig:
    k = KimiK26Config
    return KimiRefConfig(
        vocab_size=k.VOCAB_SIZE,
        hidden_size=k.EMB_SIZE,
        intermediate_size=k.INTERMEDIATE_SIZE,
        moe_intermediate_size=k.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=k.NUM_LAYERS,
        num_attention_heads=k.NUM_ATTENTION_HEADS,
        num_key_value_heads=k.NUM_KEY_VALUE_HEADS,
        q_lora_rank=k.Q_LORA_RANK,
        kv_lora_rank=k.KV_LORA_RANK,
        qk_nope_head_dim=k.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=k.QK_ROPE_HEAD_DIM,
        v_head_dim=k.V_HEAD_DIM,
        max_position_embeddings=k.MAX_POSITION_EMBEDDINGS,
        rope_theta=k.ROPE_THETA,
        rope_scaling={
            "type": "yarn",
            "factor": k.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": k.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": k.ROPE_SCALING_BETA_FAST,
            "beta_slow": k.ROPE_SCALING_BETA_SLOW,
            "mscale": k.ROPE_SCALING_MSCALE,
            "mscale_all_dim": k.ROPE_SCALING_MSCALE_ALL_DIM,
        },
        rms_norm_eps=k.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=k.NUM_DENSE_LAYERS,
        n_routed_experts=k.NUM_ROUTED_EXPERTS,
        n_shared_experts=k.NUM_SHARED_EXPERTS,
        num_experts_per_tok=k.NUM_EXPERTS_PER_TOKEN,
        n_group=k.NUM_EXPERT_GROUPS,
        topk_group=k.NUM_LIMITED_GROUPS,
        routed_scaling_factor=k.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


DSV3 = ModelVariant(
    name="dsv3",
    gate=GateRouting(8, 4, 2.5),
    build_reference_config=_build_dsv3_reference_config,
    reference_moe_cls=DSv3RefMoE,
    reference_attention_cls=DSv3RefAttention,
)

KIMI = ModelVariant(
    name="kimi",
    gate=GateRouting(1, 1, 2.827),
    build_reference_config=_build_kimi_reference_config,
    reference_moe_cls=KimiRefMoE,
    reference_attention_cls=KimiRefAttention,
    supported_meshes=frozenset({(8, 4)}),
    required_gate_fallback_mode=GateComputeMode.HOST_ALL,
    supports_pretrained=False,
)

MODEL_VARIANTS = {v.name: v for v in [DSV3, KIMI]}
