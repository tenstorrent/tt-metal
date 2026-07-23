# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model config dataclasses + the per-model build/parse factory used by train.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, NamedTuple

import ttml
from ttml.common.utils import round_up_to_tile
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from ttml.models.deepseek.flops import calculate_flops_per_token as _deepseek_flops
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig
from ttml.models.llama.flops import calculate_flops_per_token as _llama_flops
from ttml.models.nanogpt import NanoGPT, NanoGPTConfig, NanoGPTExperimentalConfig, create_nanogpt
from ttml.models.nanogpt.flops import calculate_flops_per_token as _gpt2_flops
from ttml.models.qwen3 import Qwen3, Qwen3Config, Qwen3RopeScalingConfig
from ttml.models.qwen3.flops import calculate_flops_per_token as _qwen3_flops

Model = NanoGPT | Llama | DeepSeek | Qwen3

FLOPS_REGISTRY: dict[str, Callable] = {
    "deepseek": _deepseek_flops,
    "gpt2": _gpt2_flops,
    "llama": _llama_flops,
    "qwen3": _qwen3_flops,
}


# Per-model spec dataclasses. Each is set as `ModelConfig.spec` by the
# corresponding parser in MODEL_ADAPTERS and read by the corresponding builder.


@dataclass
class _GPT2Spec:
    bias: bool = True
    positional_embedding_type: Literal["trainable", "fixed"] = "trainable"
    use_composite_layernorm: bool = False


@dataclass
class _LlamaSpec:
    num_groups: int = 3
    theta: float = 500000.0
    intermediate_dim: int | None = None
    scaling_factor: float = 0.0
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0


@dataclass
class MoEParams:
    """DeepSeek MoE expert/routing hyperparameters — the YAML ``moe_hyperparam`` block."""

    moe_inter_dim: int = 256
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 2
    n_limited_groups: int = 1
    score_func: str = "sigmoid"
    route_scale: float = 2.5


@dataclass
class _DeepSeekSpec:
    theta: float = 10000.0
    inter_dim: int | None = None
    n_dense_layers: int = 2
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    # MoE FFN variant (applies to layers >= n_dense_layers):
    #   dense     — on-device masked experts, reference/debug path (no grouping)
    #   sparse_ep — moe_group/ungroup sparse dispatch with expert-parallel experts
    # sparse_ep partitions the expert list across the "tp" axis (full-model TP) or
    # the mesh axis named by device_config.moe_axis; with no such axis (single chip)
    # it degenerates to single-device sparse (EP size 1).
    moe_type: Literal["dense", "sparse_ep"] = "sparse_ep"
    moe_hyperparam: MoEParams = field(default_factory=MoEParams)
    # Resolved MoE-parallel mesh axis name, filled from device_config.moe_axis in
    # train.py:main() (not parsed from YAML). None => no MoE-only TP axis.
    moe_axis_name: str | None = None


@dataclass
class _Qwen3Spec:
    num_groups: int = 3
    theta: float = 1000000.0
    intermediate_dim: int | None = None
    head_dim: int | None = None
    attention_bias: bool = False
    scaling_factor: float = 0.0
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0


ModelSpec = _GPT2Spec | _LlamaSpec | _DeepSeekSpec | _Qwen3Spec


@dataclass
class ModelConfig:
    """Common model fields + a `spec` set to one of the per-model spec dataclasses."""

    model_type: str = "gpt2"
    model_path: str = ""
    vocab_size: int = 0
    embedding_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 6
    dropout_prob: float = 0.2
    max_sequence_length: int = 256
    runner_type: ttml.models.RunnerType = ttml.models.RunnerType.Default
    weight_tying: ttml.models.WeightTyingType = ttml.models.WeightTyingType.Disabled
    spec: ModelSpec | None = None


# Qwen3 published RMS-norm epsilon.
_QWEN3_RMS_NORM_EPS = 1e-6


def _default_mlp_inter_dim(embedding_dim: int) -> int:
    """Default MLP intermediate dim: 8/3 × embedding_dim, rounded up to a 256-multiple."""
    return ((4 * embedding_dim * 2) // 3 + 255) // 256 * 256


def _parse_gpt2(tc: dict) -> _GPT2Spec:
    spec = _GPT2Spec()
    spec.bias = tc.get("bias", spec.bias)
    spec.positional_embedding_type = tc.get("positional_embedding_type", spec.positional_embedding_type)
    if "experimental" in tc:
        exp = tc["experimental"]
        spec.use_composite_layernorm = exp.get("use_composite_layernorm", spec.use_composite_layernorm)
    return spec


def _build_gpt2(cfg: ModelConfig, use_tp: bool) -> Model:
    if use_tp:
        raise ValueError("model_type=gpt2 has no TP path; use model_type=llama for DP+TP")
    assert isinstance(cfg.spec, _GPT2Spec)
    spec = cfg.spec
    exp = NanoGPTExperimentalConfig(use_composite_layernorm=spec.use_composite_layernorm)
    return create_nanogpt(
        NanoGPTConfig(
            vocab_size=round_up_to_tile(cfg.vocab_size, 32),
            block_size=cfg.max_sequence_length,
            n_embd=cfg.embedding_dim,
            n_layer=cfg.num_blocks,
            n_head=cfg.num_heads,
            dropout=cfg.dropout_prob,
            bias=spec.bias,
            runner_type=cfg.runner_type,
            weight_tying=cfg.weight_tying,
            positional_embedding_type=spec.positional_embedding_type,
            experimental=exp,
        )
    )


def _parse_llama(tc: dict) -> _LlamaSpec:
    spec = _LlamaSpec()
    spec.num_groups = tc.get("num_groups", spec.num_groups)
    spec.theta = tc.get("theta", spec.theta)
    spec.intermediate_dim = tc.get("intermediate_dim", spec.intermediate_dim)
    if "rope_scaling" in tc:
        rope = tc["rope_scaling"]
        spec.scaling_factor = rope.get("scaling_factor", spec.scaling_factor)
        spec.high_freq_factor = rope.get("high_freq_factor", spec.high_freq_factor)
        spec.low_freq_factor = rope.get("low_freq_factor", spec.low_freq_factor)
        spec.original_context_length = rope.get("original_context_length", spec.original_context_length)
    return spec


def _build_llama(cfg: ModelConfig, use_tp: bool) -> Model:
    assert isinstance(cfg.spec, _LlamaSpec)
    spec = cfg.spec
    if spec.num_groups <= 0:
        raise ValueError("num_groups must be a positive integer")
    if cfg.num_heads % spec.num_groups != 0:
        raise ValueError("num_heads must be divisible by num_groups")
    return Llama(
        LlamaConfig(
            hidden_size=cfg.embedding_dim,
            intermediate_size=spec.intermediate_dim,
            num_hidden_layers=cfg.num_blocks,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=spec.num_groups,
            vocab_size=round_up_to_tile(cfg.vocab_size, 32),
            max_position_embeddings=cfg.max_sequence_length,
            rope_theta=spec.theta,
            attention_dropout=cfg.dropout_prob,
            mlp_dropout=cfg.dropout_prob,
            runner_type=cfg.runner_type,
            weight_tying=cfg.weight_tying,
            rope_scaling=LlamaRopeScalingConfig(
                scaling_factor=spec.scaling_factor,
                high_freq_factor=spec.high_freq_factor,
                low_freq_factor=spec.low_freq_factor,
                original_context_length=spec.original_context_length,
            ),
            use_tp=use_tp,
        )
    )


def _parse_deepseek(tc: dict) -> _DeepSeekSpec:
    spec = _DeepSeekSpec()
    spec.theta = tc.get("theta", spec.theta)
    spec.inter_dim = tc.get("inter_dim", spec.inter_dim)
    spec.n_dense_layers = tc.get("n_dense_layers", spec.n_dense_layers)
    spec.q_lora_rank = tc.get("q_lora_rank", spec.q_lora_rank)
    spec.kv_lora_rank = tc.get("kv_lora_rank", spec.kv_lora_rank)
    spec.qk_nope_head_dim = tc.get("qk_nope_head_dim", spec.qk_nope_head_dim)
    spec.qk_rope_head_dim = tc.get("qk_rope_head_dim", spec.qk_rope_head_dim)
    spec.v_head_dim = tc.get("v_head_dim", spec.v_head_dim)

    spec.moe_type = tc.get("moe_type", spec.moe_type)
    # Expert/routing knobs live under `moe_hyperparam`; fall back to flat top-level
    # keys so legacy configs keep parsing.
    moe = tc.get("moe_hyperparam", tc)
    h = spec.moe_hyperparam
    h.moe_inter_dim = moe.get("moe_inter_dim", h.moe_inter_dim)
    h.n_routed_experts = moe.get("n_routed_experts", h.n_routed_experts)
    h.n_shared_experts = moe.get("n_shared_experts", h.n_shared_experts)
    h.n_activated_experts = moe.get("n_activated_experts", h.n_activated_experts)
    h.n_expert_groups = moe.get("n_expert_groups", h.n_expert_groups)
    h.n_limited_groups = moe.get("n_limited_groups", h.n_limited_groups)
    h.score_func = moe.get("score_func", h.score_func)
    h.route_scale = moe.get("route_scale", h.route_scale)
    return spec


_MOE_TYPES = ("dense", "sparse_ep")


def _build_deepseek(cfg: ModelConfig, use_tp: bool) -> Model:
    # DeepSeek integrates with the named-mesh TP path (MLA, dense MLP, LM head, and — under
    # full-model TP — sparse MoE all shard across the "tp" axis). moe_type selects the MoE FFN
    # variant; sparse_ep additionally partitions experts across a mesh axis.
    assert isinstance(cfg.spec, _DeepSeekSpec)
    spec = cfg.spec
    if spec.moe_type not in _MOE_TYPES:
        raise ValueError(f"Unknown moe_type={spec.moe_type!r}; expected one of {list(_MOE_TYPES)}")

    # Resolve the expert-sharding axis for sparse_ep: "tp" under full-model TP, else
    # the axis resolved from device_config.moe_axis (set on the spec in main()). With no usable
    # axis the library falls back to single-chip SparseMoE, so a missing axis is not an error.
    moe_axis_name = None
    if spec.moe_type == "sparse_ep":
        moe_axis_name = "tp" if use_tp else spec.moe_axis_name

    h = spec.moe_hyperparam
    inter_dim = spec.inter_dim or _default_mlp_inter_dim(cfg.embedding_dim)
    return DeepSeek(
        DeepSeekConfig(
            vocab_size=round_up_to_tile(cfg.vocab_size, 32),
            dim=cfg.embedding_dim,
            inter_dim=inter_dim,
            moe_inter_dim=h.moe_inter_dim,
            n_layers=cfg.num_blocks,
            n_dense_layers=spec.n_dense_layers,
            n_heads=cfg.num_heads,
            n_routed_experts=h.n_routed_experts,
            n_shared_experts=h.n_shared_experts,
            n_activated_experts=h.n_activated_experts,
            n_expert_groups=h.n_expert_groups,
            n_limited_groups=h.n_limited_groups,
            score_func=h.score_func,
            route_scale=h.route_scale,
            moe_type=spec.moe_type,
            q_lora_rank=spec.q_lora_rank,
            kv_lora_rank=spec.kv_lora_rank,
            qk_nope_head_dim=spec.qk_nope_head_dim,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            v_head_dim=spec.v_head_dim,
            max_seq_len=cfg.max_sequence_length,
            rope_theta=spec.theta,
            runner_type=cfg.runner_type,
            moe_axis_name=moe_axis_name,
            use_tp=use_tp,
        )
    )


def _parse_qwen3(tc: dict) -> _Qwen3Spec:
    spec = _Qwen3Spec()
    spec.num_groups = tc.get("num_groups", spec.num_groups)
    spec.theta = tc.get("theta", spec.theta)
    spec.intermediate_dim = tc.get("intermediate_dim", spec.intermediate_dim)
    spec.head_dim = tc.get("head_dim", spec.head_dim)
    spec.attention_bias = tc.get("attention_bias", spec.attention_bias)
    if "rope_scaling" in tc:
        rope = tc["rope_scaling"]
        spec.scaling_factor = rope.get("scaling_factor", spec.scaling_factor)
        spec.high_freq_factor = rope.get("high_freq_factor", spec.high_freq_factor)
        spec.low_freq_factor = rope.get("low_freq_factor", spec.low_freq_factor)
        spec.original_context_length = rope.get("original_context_length", spec.original_context_length)
    return spec


def _build_qwen3(cfg: ModelConfig, use_tp: bool) -> Model:
    if use_tp:
        raise ValueError("model_type=qwen3 has no TP path; use model_type=llama for DP+TP")
    assert isinstance(cfg.spec, _Qwen3Spec)
    spec = cfg.spec
    head_dim = spec.head_dim or cfg.embedding_dim // cfg.num_heads
    intermediate = spec.intermediate_dim or _default_mlp_inter_dim(cfg.embedding_dim)
    return Qwen3(
        Qwen3Config(
            hidden_size=cfg.embedding_dim,
            intermediate_size=intermediate,
            num_hidden_layers=cfg.num_blocks,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=spec.num_groups,
            head_dim=head_dim,
            vocab_size=round_up_to_tile(cfg.vocab_size, 32),
            max_position_embeddings=cfg.max_sequence_length,
            rms_norm_eps=_QWEN3_RMS_NORM_EPS,
            attention_bias=spec.attention_bias,
            attention_dropout=cfg.dropout_prob,
            rope_theta=spec.theta,
            runner_type=cfg.runner_type,
            weight_tying=cfg.weight_tying,
            rope_scaling=Qwen3RopeScalingConfig(
                scaling_factor=spec.scaling_factor,
                high_freq_factor=spec.high_freq_factor,
                low_freq_factor=spec.low_freq_factor,
                original_context_length=spec.original_context_length,
            ),
        )
    )


ParseFn = Callable[[dict], ModelSpec]
BuildFn = Callable[[ModelConfig, bool], Model]


class ModelAdapter(NamedTuple):
    # parse builds a spec from YAML; build consumes `cfg` + `cfg.spec`
    # and returns the constructed model.
    parse: ParseFn
    build: BuildFn


MODEL_ADAPTERS: dict[str, ModelAdapter] = {
    "gpt2": ModelAdapter(_parse_gpt2, _build_gpt2),
    "llama": ModelAdapter(_parse_llama, _build_llama),
    "deepseek": ModelAdapter(_parse_deepseek, _build_deepseek),
    "qwen3": ModelAdapter(_parse_qwen3, _build_qwen3),
}


def parse_model_config(yaml_config: dict) -> ModelConfig:
    """Parse YAML into a ModelConfig: common fields directly, model-specific via MODEL_ADAPTERS."""
    transformer_config = yaml_config.get("transformer_config", {})
    cfg = ModelConfig()

    cfg.model_type = transformer_config.get("model_type", cfg.model_type)
    cfg.model_path = transformer_config.get("model_path", cfg.model_path)
    cfg.vocab_size = transformer_config.get("vocab_size", cfg.vocab_size)
    cfg.embedding_dim = transformer_config.get("embedding_dim", cfg.embedding_dim)
    cfg.num_blocks = transformer_config.get("num_blocks", cfg.num_blocks)
    cfg.num_heads = transformer_config.get("num_heads", cfg.num_heads)
    cfg.dropout_prob = transformer_config.get("dropout_prob", cfg.dropout_prob)
    cfg.max_sequence_length = transformer_config.get("max_sequence_length", cfg.max_sequence_length)

    if "runner_type" in transformer_config:
        cfg.runner_type = ttml.models.RunnerType.from_string(transformer_config["runner_type"])
    if "weight_tying" in transformer_config:
        cfg.weight_tying = ttml.models.WeightTyingType.from_string(transformer_config["weight_tying"])

    adapter = MODEL_ADAPTERS.get(cfg.model_type)
    if adapter is None:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
    cfg.spec = adapter.parse(transformer_config)
    return cfg


def create_model(cfg: ModelConfig, use_tp: bool = False) -> Model:
    """Dispatch on `cfg.model_type` to the registered builder.

    `cfg.vocab_size` is the raw (unpadded) vocab; each builder tile-pads it to a 32-multiple
    for the model, since the embedding/LM-head weights must be tile-aligned.
    """
    adapter = MODEL_ADAPTERS.get(cfg.model_type)
    if adapter is None:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
    return adapter.build(cfg, use_tp)


def instantiate_model_from_config(cfg: ModelConfig, lazy_init: bool, *, use_tp: bool = False) -> Model:
    """Create the model, deferring Parameter allocation when ``lazy_init`` is True.

    With ``lazy_init`` the returned model holds ``TensorMetadata`` for every parameter; the caller
    must call ``ttml.materialize_module(model)`` after any pre-materialize transforms (e.g.
    ``ttml.fsdp.fully_shard``, which rewrites each lazy param's mapper so materialize allocates the
    weights already-sharded). With ``lazy_init=False`` parameters are allocated eagerly.
    """
    if lazy_init:
        with ttml.lazy_init():
            return create_model(cfg, use_tp=use_tp)
    return create_model(cfg, use_tp=use_tp)
