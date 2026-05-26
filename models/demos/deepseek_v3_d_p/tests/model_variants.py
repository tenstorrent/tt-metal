# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-variant test configurations for the DeepSeek V3 architecture family.

Adding a new variant: subclass `ModelVariant`, implement
`build_reference_config()` to return a populated `PretrainedConfig`,
instantiate `<NAME> = <SubclassName>(...)`, and append it to
`MODEL_VARIANTS` at the bottom. Wire `reference_*_cls` only if you also
vendor that variant's upstream reference into `deepseek_v3_d_p/reference/`.
"""

from abc import ABC
from typing import FrozenSet, Optional, Tuple

from transformers.configuration_utils import PretrainedConfig

from models.demos.deepseek_v3_d_p.reference.kimi_k26.configuration_deepseek import DeepseekV3Config as KimiRefConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k26.kimi_k26_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3Attention as KimiRefAttention
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3MoE as KimiRefMoE
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode


class ModelVariant(ABC):
    """One variant of the DeepSeek V3 architecture family.

    Subclasses implement `build_reference_config()` returning a populated
    `PretrainedConfig`. The config is built once at construction and is the
    single source of truth for variant model params; read it via
    `get_config()`. `reference_*_cls` must be set for the upstream HF
    cross-check to actually run; without it, the config is still used to
    parametrize TT/torch constructors.
    """

    def __init__(
        self,
        *,
        name: str,
        reference_moe_cls: Optional[type] = None,
        reference_attention_cls: Optional[type] = None,
        moe_pcc_threshold: float = 0.999,
        mla_pcc_threshold: float = 0.999,
        supported_meshes: Optional[FrozenSet[Tuple[int, int]]] = None,
        required_gate_fallback_mode: Optional[GateComputeMode] = None,
        supports_pretrained: bool = True,
    ) -> None:
        self.name = name
        self.reference_moe_cls = reference_moe_cls
        self.reference_attention_cls = reference_attention_cls
        self.moe_pcc_threshold = moe_pcc_threshold
        self.mla_pcc_threshold = mla_pcc_threshold
        self.supported_meshes = supported_meshes
        self.required_gate_fallback_mode = required_gate_fallback_mode
        self.supports_pretrained = supports_pretrained
        self._config: Optional[PretrainedConfig] = None

    def build_reference_config(self) -> Optional[PretrainedConfig]:
        """Construct an HF config from constants. Default: None (variant
        overrides `get_config` to source its config elsewhere)."""
        return None

    def get_config(self, request) -> PretrainedConfig:
        """Resolve and cache the variant's HF reference config. Default
        sources from `build_reference_config()`; variants that load from
        disk override this and read a fixture off `request`."""
        if self._config is None:
            self._config = self.build_reference_config()
        return self._config


# ---------------------------------------------------------------------------
# Variants — one subclass per model
# ---------------------------------------------------------------------------


class DSv3Variant(ModelVariant):
    def get_config(self, request) -> PretrainedConfig:
        if self._config is None:
            self._config = request.getfixturevalue("config_only")
        return self._config


class KimiVariant(ModelVariant):
    def build_reference_config(self) -> KimiRefConfig:
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


DSV3 = DSv3Variant(name="dsv3")

KIMI = KimiVariant(
    name="kimi",
    reference_moe_cls=KimiRefMoE,
    reference_attention_cls=KimiRefAttention,
    supported_meshes=frozenset({(8, 4)}),
    required_gate_fallback_mode=GateComputeMode.HOST_ALL,
    supports_pretrained=False,
    mla_pcc_threshold=0.995,
    moe_pcc_threshold=0.989,
)

MODEL_VARIANTS = {v.name: v for v in [DSV3, KIMI]}
