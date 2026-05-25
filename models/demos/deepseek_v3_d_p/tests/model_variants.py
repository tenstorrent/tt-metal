# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-variant test configurations for the DeepSeek V3 architecture family.

DSv3 and Kimi K2.6 share the DeepSeek V3 architecture (Kimi ships with
`architectures: ["DeepseekV3ForCausalLM"]`); only hyperparameters differ.
This module bundles those differences so that a single parametrized test
body covers both variants — see test_ttnn_moe.py and test_mla.py.

Public surface:
- ModelVariant (ABC), GateRouting        — variant base class + gate routing dataclass
- ModelVariant.apply_gate_overrides_to_*  — runtime gate-config patch on TT / Torch MoE
- MODEL_VARIANTS                         — {name: ModelVariant}

Upstream reference runners (`run_reference_moe`, `run_reference_mla`) live
in `reference_runners.py` — variants are a pure registry here.

Adding a new variant: subclass `ModelVariant`, implement
`build_reference_config()` (return `None` if no HF cross-check), instantiate
`<NAME> = <SubclassName>(...)`, and append it to `MODEL_VARIANTS` at the
bottom. Wire `reference_*_cls` only if you also vendor that variant's
upstream reference into `d_p/reference/` (Kimi does this; DSv3 doesn't).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from transformers.configuration_utils import PretrainedConfig

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config as DSv3RefConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26.configuration_deepseek import DeepseekV3Config as KimiRefConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k26.kimi_k26_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3Attention as KimiRefAttention
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3MoE as KimiRefMoE
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode


@dataclass(frozen=True)
class GateRouting:
    """MoE gate routing knobs. All fields mandatory — defaults would silently
    encode one variant's values and create a hidden source of truth."""

    n_expert_groups: int
    n_limited_groups: int
    route_scale: float


class ModelVariant(ABC):
    """One variant of the DeepSeek V3 architecture family.

    `name` + `gate` + the torch reference path (TorchMoe / create_mla_reference)
    drive the always-on PCC comparison with this variant's hyperparameters.

    Subclasses implement `build_reference_config()` — return a
    `PretrainedConfig` to enable the upstream HF cross-check, or `None` to
    skip it. `reference_*_cls` must be set on the variant for the cross-check
    to run.

    Test-environment constraints (`supported_meshes`, `required_gate_fallback_mode`,
    `supports_pretrained`) let tests gate cases generically — no variant-name
    conditionals required in the test body.
    """

    def __init__(
        self,
        *,
        name: str,
        gate: GateRouting,
        reference_moe_cls: Optional[type] = None,
        reference_attention_cls: Optional[type] = None,
        moe_pcc_threshold: float = 0.96,
        mla_pcc_threshold: float = 0.98,
        supported_meshes: Optional[FrozenSet[Tuple[int, int]]] = None,
        required_gate_fallback_mode: Optional[GateComputeMode] = None,
        supports_pretrained: bool = True,
    ) -> None:
        self.name = name
        self.gate = gate
        self.reference_moe_cls = reference_moe_cls
        self.reference_attention_cls = reference_attention_cls
        self.moe_pcc_threshold = moe_pcc_threshold
        self.mla_pcc_threshold = mla_pcc_threshold
        self.supported_meshes = supported_meshes
        self.required_gate_fallback_mode = required_gate_fallback_mode
        self.supports_pretrained = supports_pretrained

    @abstractmethod
    def build_reference_config(self) -> Optional[PretrainedConfig]:
        """Variant-specific HF reference config for the cross-check, or None to skip.
        Subclasses must override."""

    def apply_gate_overrides_to_tt_moe(self, tt_moe) -> None:
        """Patch gate routing on a `TtMoe` post-construction.

        Why post-construction: TtMoEGateConfig bakes DSv3 defaults at init and
        doesn't accept variant routing as a parameter — this is the test-side
        bridge until module configs are taken via constructor args (Marko's
        Stage 0).
        Why this is safe today: the host gate path re-reads `self.config.*`
        at forward time. The init-time `reference_model` snapshot is kept in
        sync below so any future code path that consults it stays consistent.
        """
        g = self.gate
        cfg = tt_moe.gate.config
        cfg.n_expert_groups = g.n_expert_groups
        cfg.n_limited_groups = g.n_limited_groups
        cfg.route_scale = g.route_scale
        ref_cfg = getattr(tt_moe.gate, "ref_config", None)
        if ref_cfg is not None:
            ref_cfg.n_group = g.n_expert_groups
            ref_cfg.topk_group = g.n_limited_groups
            ref_cfg.routed_scaling_factor = g.route_scale
        ref_model = getattr(tt_moe.gate, "reference_model", None)
        if ref_model is not None:
            ref_model.n_group = g.n_expert_groups
            ref_model.topk_group = g.n_limited_groups
            ref_model.routed_scaling_factor = g.route_scale

    def apply_gate_overrides_to_torch_moe(self, torch_moe) -> None:
        """Patch gate routing on a `TorchMoe` post-construction. See
        `apply_gate_overrides_to_tt_moe` for the post-construction rationale."""
        g = self.gate
        gate = torch_moe.gate
        gate.n_group = g.n_expert_groups
        gate.topk_group = g.n_limited_groups
        gate.routed_scaling_factor = g.route_scale


# ---------------------------------------------------------------------------
# Variants — one subclass per model, implementing build_reference_config().
# ---------------------------------------------------------------------------


class DSv3Variant(ModelVariant):
    def build_reference_config(self) -> DSv3RefConfig:
        c = DeepSeekV3Config
        return DSv3RefConfig(
            hidden_size=c.EMB_SIZE,
            num_attention_heads=c.NUM_ATTENTION_HEADS,
            num_key_value_heads=c.NUM_ATTENTION_HEADS,
            q_lora_rank=c.Q_LORA_RANK,
            kv_lora_rank=c.KV_LORA_RANK,
            qk_nope_head_dim=c.QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=c.QK_ROPE_HEAD_DIM,
            v_head_dim=c.V_HEAD_DIM,
        )


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


DSV3 = DSv3Variant(
    name="dsv3",
    gate=GateRouting(
        n_expert_groups=DeepSeekV3Config.NUM_EXPERT_GROUPS,
        n_limited_groups=DeepSeekV3Config.NUM_LIMITED_GROUPS,
        route_scale=DeepSeekV3Config.ROUTE_SCALE,
    ),
)

KIMI = KimiVariant(
    name="kimi",
    gate=GateRouting(
        n_expert_groups=KimiK26Config.NUM_EXPERT_GROUPS,
        n_limited_groups=KimiK26Config.NUM_LIMITED_GROUPS,
        route_scale=KimiK26Config.ROUTE_SCALE,
    ),
    reference_moe_cls=KimiRefMoE,
    reference_attention_cls=KimiRefAttention,
    supported_meshes=frozenset({(8, 4)}),
    required_gate_fallback_mode=GateComputeMode.HOST_ALL,
    supports_pretrained=False,
)

MODEL_VARIANTS = {v.name: v for v in [DSV3, KIMI]}
