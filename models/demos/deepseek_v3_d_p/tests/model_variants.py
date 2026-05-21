# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-variant test configurations for DeepSeek V3 and Kimi K2.6.

DSv3 and Kimi K2.6 share the DeepSeek V3 architecture (Kimi ships with
`architectures: ["DeepseekV3ForCausalLM"]`); only hyperparameters differ.
This module bundles those differences so that a single parametrized test
body covers both variants — see test_ttnn_moe.py and test_mla.py.

Pattern mirrors models/tt_dit/tests/unit/test_ring_joint_attention.py:625-690.

Public surface:
- MoeModelConfig, MlaModelConfig                  — variant dataclass
- apply_gate_overrides_moe                        — runtime gate-config patch
- MOE_MODEL_CONFIGS, MLA_MODEL_CONFIGS            — model-name → config dicts

Each variant's HF reference (model class + config builder) is sourced from
that variant's own folder:
- DSv3:  models/demos/deepseek_v3/reference/
- Kimi:  models/experimental/kimi_k26/reference/
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config as DSv3HfConfig
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention as DSv3HfAttention
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE as DSv3HfMoE
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26_config import KimiK26Config
from models.experimental.kimi_k26.reference.configuration_deepseek import DeepseekV3Config as KimiHfConfig
from models.experimental.kimi_k26.reference.modeling_deepseek import DeepseekV3Attention as KimiHfAttention
from models.experimental.kimi_k26.reference.modeling_deepseek import DeepseekV3MoE as KimiHfMoE

# ---------------------------------------------------------------------------
# MoE variant config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeModelConfig:
    """Per-variant MoE test configuration.

    Gate-routing knobs (`n_expert_groups`, `n_limited_groups`, `route_scale`)
    are patched onto TtMoe and TorchMoe gates at runtime so the same test body
    covers both variants — DSv3 values are a no-op patch. `hf_ref` is the
    variant-specific upstream HF reference (model class + config builder from
    that variant's own folder); the test runs it as an extra PCC cross-check.
    """

    n_expert_groups: int
    n_limited_groups: int
    route_scale: float
    hf_ref: Optional[Callable] = None
    hf_pcc_threshold: float = 0.96


def apply_gate_overrides_moe(tt_moe, torch_moe, model_config: MoeModelConfig) -> None:
    """Patch gate config on TT and Torch MoE references.

    DSv3 values are a no-op patch. Either side may be None.
    """
    if tt_moe is not None:
        cfg = tt_moe.gate.config
        cfg.n_expert_groups = model_config.n_expert_groups
        cfg.n_limited_groups = model_config.n_limited_groups
        cfg.route_scale = model_config.route_scale
    if torch_moe is not None:
        g = torch_moe.gate
        g.n_group = model_config.n_expert_groups
        g.topk_group = model_config.n_limited_groups
        g.routed_scaling_factor = model_config.route_scale


def _build_dsv3_hf_config_moe() -> DSv3HfConfig:
    return DSv3HfConfig(
        vocab_size=DeepSeekV3Config.VOCAB_SIZE,
        hidden_size=DeepSeekV3Config.EMB_SIZE,
        intermediate_size=DeepSeekV3Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=DeepSeekV3Config.NUM_LAYERS,
        num_attention_heads=DeepSeekV3Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=DeepSeekV3Config.NUM_ATTENTION_HEADS,
        q_lora_rank=DeepSeekV3Config.Q_LORA_RANK,
        kv_lora_rank=DeepSeekV3Config.KV_LORA_RANK,
        qk_nope_head_dim=DeepSeekV3Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=DeepSeekV3Config.QK_ROPE_HEAD_DIM,
        v_head_dim=DeepSeekV3Config.V_HEAD_DIM,
        rms_norm_eps=DeepSeekV3Config.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=DeepSeekV3Config.NUM_DENSE_LAYERS,
        n_routed_experts=DeepSeekV3Config.NUM_ROUTED_EXPERTS,
        n_shared_experts=DeepSeekV3Config.NUM_SHARED_EXPERTS,
        num_experts_per_tok=DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN,
        n_group=DeepSeekV3Config.NUM_EXPERT_GROUPS,
        topk_group=DeepSeekV3Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=DeepSeekV3Config.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


def _build_kimi_hf_config_moe() -> KimiHfConfig:
    return KimiHfConfig(
        vocab_size=KimiK26Config.VOCAB_SIZE,
        hidden_size=KimiK26Config.EMB_SIZE,
        intermediate_size=KimiK26Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=KimiK26Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=KimiK26Config.NUM_LAYERS,
        num_attention_heads=KimiK26Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=KimiK26Config.NUM_KEY_VALUE_HEADS,
        q_lora_rank=KimiK26Config.Q_LORA_RANK,
        kv_lora_rank=KimiK26Config.KV_LORA_RANK,
        qk_nope_head_dim=KimiK26Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=KimiK26Config.QK_ROPE_HEAD_DIM,
        v_head_dim=KimiK26Config.V_HEAD_DIM,
        max_position_embeddings=KimiK26Config.MAX_POSITION_EMBEDDINGS,
        rope_theta=KimiK26Config.ROPE_THETA,
        rope_scaling={
            "type": "yarn",
            "factor": KimiK26Config.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": KimiK26Config.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": KimiK26Config.ROPE_SCALING_BETA_FAST,
            "beta_slow": KimiK26Config.ROPE_SCALING_BETA_SLOW,
            "mscale": KimiK26Config.ROPE_SCALING_MSCALE,
            "mscale_all_dim": KimiK26Config.ROPE_SCALING_MSCALE_ALL_DIM,
        },
        rms_norm_eps=KimiK26Config.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=KimiK26Config.NUM_DENSE_LAYERS,
        n_routed_experts=KimiK26Config.NUM_ROUTED_EXPERTS,
        n_shared_experts=KimiK26Config.NUM_SHARED_EXPERTS,
        num_experts_per_tok=KimiK26Config.NUM_EXPERTS_PER_TOKEN,
        n_group=KimiK26Config.NUM_EXPERT_GROUPS,
        topk_group=KimiK26Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=KimiK26Config.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


def _pack_hf_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights) -> dict:
    """Repack random TT-style weights into the HF DeepseekV3MoE state-dict layout."""
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


def _run_hf_moe(model_cls, hf_config, gate_weights, routed_expert_weights, shared_expert_weights, x):
    moe = model_cls(hf_config)
    moe.load_state_dict(
        _pack_hf_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights), strict=False
    )
    moe = moe.eval().to(torch.bfloat16)
    with torch.no_grad():
        return moe(x.to(torch.bfloat16))


def _dsv3_hf_ref_moe(*, gate_weights, routed_expert_weights, shared_expert_weights, x) -> torch.Tensor:
    return _run_hf_moe(
        DSv3HfMoE, _build_dsv3_hf_config_moe(), gate_weights, routed_expert_weights, shared_expert_weights, x
    )


def _kimi_hf_ref_moe(*, gate_weights, routed_expert_weights, shared_expert_weights, x) -> torch.Tensor:
    return _run_hf_moe(
        KimiHfMoE, _build_kimi_hf_config_moe(), gate_weights, routed_expert_weights, shared_expert_weights, x
    )


MOE_MODEL_CONFIGS: dict[str, MoeModelConfig] = {
    "dsv3": MoeModelConfig(n_expert_groups=8, n_limited_groups=4, route_scale=2.5, hf_ref=_dsv3_hf_ref_moe),
    "kimi": MoeModelConfig(n_expert_groups=1, n_limited_groups=1, route_scale=2.827, hf_ref=_kimi_hf_ref_moe),
}


# ---------------------------------------------------------------------------
# MLA variant config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MlaModelConfig:
    """Per-variant MLA test configuration.

    `config_builder=None` means "use the DSv3 `config_only` fixture" (existing
    behavior). When set (e.g. Kimi), it constructs a DeepseekV3Config populated
    with that variant's hyperparameters; that config drives weight shapes,
    ttMLA, and `create_mla_reference`. `hf_ref` is the variant-specific
    upstream HF cross-check (model class + config builder from that variant's
    folder).
    """

    name: str
    config_builder: Optional[Callable] = None
    hf_ref: Optional[Callable] = None
    hf_pcc_threshold: float = 0.98


def _build_dsv3_hf_config_mla() -> DSv3HfConfig:
    """HF DSv3HfConfig populated with DSv3 MLA hyperparameters."""
    return DSv3HfConfig(
        vocab_size=DeepSeekV3Config.VOCAB_SIZE,
        hidden_size=DeepSeekV3Config.EMB_SIZE,
        intermediate_size=DeepSeekV3Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=DeepSeekV3Config.NUM_LAYERS,
        num_attention_heads=DeepSeekV3Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=DeepSeekV3Config.NUM_ATTENTION_HEADS,
        q_lora_rank=DeepSeekV3Config.Q_LORA_RANK,
        kv_lora_rank=DeepSeekV3Config.KV_LORA_RANK,
        qk_nope_head_dim=DeepSeekV3Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=DeepSeekV3Config.QK_ROPE_HEAD_DIM,
        v_head_dim=DeepSeekV3Config.V_HEAD_DIM,
        rms_norm_eps=DeepSeekV3Config.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
    )


def _build_kimi_hf_config_mla() -> DSv3HfConfig:
    """HF config populated with Kimi K2.6 hyperparameters.

    NOTE: instantiates the DSv3 vendor's config class (not Kimi's) — see
    the asymmetry comment on MLA in the change log. Functional for MLA
    because DeepseekV3Attention only reads MLA-relevant fields that exist
    on both vendor configs.
    """
    return DSv3HfConfig(
        vocab_size=KimiK26Config.VOCAB_SIZE,
        hidden_size=KimiK26Config.EMB_SIZE,
        intermediate_size=KimiK26Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=KimiK26Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=KimiK26Config.NUM_LAYERS,
        num_attention_heads=KimiK26Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=KimiK26Config.NUM_KEY_VALUE_HEADS,
        q_lora_rank=KimiK26Config.Q_LORA_RANK,
        kv_lora_rank=KimiK26Config.KV_LORA_RANK,
        qk_nope_head_dim=KimiK26Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=KimiK26Config.QK_ROPE_HEAD_DIM,
        v_head_dim=KimiK26Config.V_HEAD_DIM,
        max_position_embeddings=KimiK26Config.MAX_POSITION_EMBEDDINGS,
        rope_theta=KimiK26Config.ROPE_THETA,
        rope_scaling={
            "type": "yarn",
            "factor": KimiK26Config.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": KimiK26Config.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": KimiK26Config.ROPE_SCALING_BETA_FAST,
            "beta_slow": KimiK26Config.ROPE_SCALING_BETA_SLOW,
            "mscale": KimiK26Config.ROPE_SCALING_MSCALE,
            "mscale_all_dim": KimiK26Config.ROPE_SCALING_MSCALE_ALL_DIM,
        },
        rms_norm_eps=KimiK26Config.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=KimiK26Config.NUM_DENSE_LAYERS,
        n_routed_experts=KimiK26Config.NUM_ROUTED_EXPERTS,
        n_shared_experts=KimiK26Config.NUM_SHARED_EXPERTS,
        num_experts_per_tok=KimiK26Config.NUM_EXPERTS_PER_TOKEN,
        n_group=KimiK26Config.NUM_EXPERT_GROUPS,
        topk_group=KimiK26Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=KimiK26Config.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


def _run_hf_mla(model_cls, hf_config, weights, hidden_states, position_ids) -> torch.Tensor:
    """Forward upstream HF DeepseekV3Attention on CPU and return its output."""
    attn = model_cls(hf_config, layer_idx=0)
    attn.load_state_dict(weights, strict=False)
    attn = attn.eval().to(torch.bfloat16)
    _, q_len, _ = hidden_states.shape
    causal = torch.full((q_len, q_len), float("-inf"), dtype=hidden_states.dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out, _, _ = attn(
            hidden_states=hidden_states,
            attention_mask=causal,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
        )
    return out


def _dsv3_hf_ref_mla(*, weights, hidden_states, position_ids) -> torch.Tensor:
    return _run_hf_mla(DSv3HfAttention, _build_dsv3_hf_config_mla(), weights, hidden_states, position_ids)


def _kimi_hf_ref_mla(*, weights, hidden_states, position_ids) -> torch.Tensor:
    return _run_hf_mla(KimiHfAttention, _build_kimi_hf_config_mla(), weights, hidden_states, position_ids)


MLA_MODEL_CONFIGS: dict[str, MlaModelConfig] = {
    "dsv3": MlaModelConfig(name="dsv3", config_builder=None, hf_ref=_dsv3_hf_ref_mla),
    "kimi": MlaModelConfig(name="kimi", config_builder=_build_kimi_hf_config_mla, hf_ref=_kimi_hf_ref_mla),
}
