# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT/Llama/DeepSeek/Qwen3 training entry point built on `SFTTrainer`.

Two modes, selected by CLI flags:

* Training (default): build dataset + model + trainer from a YAML config and CLI
  overrides, then run.
* Inference: with both `--prompt` and `--model-path` set, load a checkpoint and
  generate text.

Old `train_nanogpt.py` callers keep working: underscore-form flags are accepted
silently, and the three renamed flags (`--num_epochs`, `--clip_grad_norm`,
`--model_save_path`) are accepted with a deprecation warning.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple

import ml_dtypes
import numpy as np
import yaml

import ttnn
from ttnn.device import is_blackhole, is_wormhole_b0
import ttml
import ttml.common.muon_optimizer
from ttml.common.config import DeviceConfig, SchedulerConfig, TrainingConfig as BaseTrainingConfig, load_config
from ttml.common.data import CharTokenizer, build_causal_mask
from ttml.common.profiler_utils import profiler_marker
from ttml.common.schedulers import SpeedrunScheduler
from ttml.common.utils import (
    create_optimizer,
    get_available_device_memory_in_bytes,
    get_tt_metal_runtime_root,
    round_up_to_tile,
    summary,
)
from ttml.datasets import Batch, InMemoryDataloader, causal_lm_collate_fn
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from ttml.models.deepseek.flops import calculate_flops_per_token as _deepseek_flops
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig
from ttml.models.llama.flops import calculate_flops_per_token as _llama_flops
from ttml.models.nanogpt import NanoGPT, NanoGPTConfig, NanoGPTExperimentalConfig, create_nanogpt
from ttml.models.nanogpt.flops import calculate_flops_per_token as _gpt2_flops
from ttml.models.qwen3 import Qwen3, Qwen3Config, Qwen3RopeScalingConfig
from ttml.models.qwen3.flops import calculate_flops_per_token as _qwen3_flops
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

import moe_activation_logger

# Side effect: registers the Muon optimizer with ttml's optimizer factory so
# YAML configs can name it.
ttml.common.muon_optimizer.register()

Model = NanoGPT | Llama | DeepSeek | Qwen3
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker

FLOPS_REGISTRY: dict[str, Callable] = {
    "deepseek": _deepseek_flops,
    "gpt2": _gpt2_flops,
    "llama": _llama_flops,
    "qwen3": _qwen3_flops,
}


# ── Config dataclasses ────────────────────────────────────────────────────────


class TrainingConfig(BaseTrainingConfig):
    """Base training config + NanoGPT-specific defaults + legacy field-name aliases."""

    def __init__(self, yaml_config: dict | None = None) -> None:
        yaml_config = yaml_config if yaml_config is not None else {}
        super().__init__(yaml_config)
        tc = yaml_config.get("training_config", {})

        self.project_name = tc.get("project_name", "tt_train_nano_gpt")
        self.data_path = tc.get("data_path", "")
        self.scheduler_type = tc.get("scheduler_type", "identity")
        self.use_clip_grad_norm = tc.get("use_clip_grad_norm", False)
        self.clip_grad_norm_max_norm = float(tc.get("clip_grad_norm_max_norm", 1.0))

        # Override base-class defaults with NanoGPT-specific values.
        self.seed = int(tc.get("seed", 5489))
        self.max_steps = int(tc.get("max_steps", 5000))

        # Legacy field names kept alive for callers that still read them.
        self.num_epochs = self.epochs
        self.model_save_interval = self.save_every


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
class _DeepSeekSpec:
    theta: float = 10000.0
    inter_dim: int | None = None
    moe_inter_dim: int = 256
    n_dense_layers: int = 2
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 2
    n_limited_groups: int = 1
    score_func: str = "sigmoid"
    route_scale: float = 2.5
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64


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


# ── Models ────────────────────────────────────────────────────────────────────

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
            vocab_size=cfg.vocab_size,
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
            vocab_size=cfg.vocab_size,
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
    spec.moe_inter_dim = tc.get("moe_inter_dim", spec.moe_inter_dim)
    spec.n_dense_layers = tc.get("n_dense_layers", spec.n_dense_layers)
    spec.n_routed_experts = tc.get("n_routed_experts", spec.n_routed_experts)
    spec.n_shared_experts = tc.get("n_shared_experts", spec.n_shared_experts)
    spec.n_activated_experts = tc.get("n_activated_experts", spec.n_activated_experts)
    spec.n_expert_groups = tc.get("n_expert_groups", spec.n_expert_groups)
    spec.n_limited_groups = tc.get("n_limited_groups", spec.n_limited_groups)
    spec.score_func = tc.get("score_func", spec.score_func)
    spec.route_scale = tc.get("route_scale", spec.route_scale)
    spec.q_lora_rank = tc.get("q_lora_rank", spec.q_lora_rank)
    spec.kv_lora_rank = tc.get("kv_lora_rank", spec.kv_lora_rank)
    spec.qk_nope_head_dim = tc.get("qk_nope_head_dim", spec.qk_nope_head_dim)
    spec.qk_rope_head_dim = tc.get("qk_rope_head_dim", spec.qk_rope_head_dim)
    spec.v_head_dim = tc.get("v_head_dim", spec.v_head_dim)
    return spec


def _build_deepseek(cfg: ModelConfig, use_tp: bool) -> Model:
    if use_tp:
        raise ValueError("model_type=deepseek has no TP path; use model_type=llama for DP+TP")
    assert isinstance(cfg.spec, _DeepSeekSpec)
    spec = cfg.spec
    inter_dim = spec.inter_dim or _default_mlp_inter_dim(cfg.embedding_dim)
    return DeepSeek(
        DeepSeekConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.embedding_dim,
            inter_dim=inter_dim,
            moe_inter_dim=spec.moe_inter_dim,
            n_layers=cfg.num_blocks,
            n_dense_layers=spec.n_dense_layers,
            n_heads=cfg.num_heads,
            n_routed_experts=spec.n_routed_experts,
            n_shared_experts=spec.n_shared_experts,
            n_activated_experts=spec.n_activated_experts,
            n_expert_groups=spec.n_expert_groups,
            n_limited_groups=spec.n_limited_groups,
            score_func=spec.score_func,
            route_scale=spec.route_scale,
            q_lora_rank=spec.q_lora_rank,
            kv_lora_rank=spec.kv_lora_rank,
            qk_nope_head_dim=spec.qk_nope_head_dim,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            v_head_dim=spec.v_head_dim,
            max_seq_len=cfg.max_sequence_length,
            rope_theta=spec.theta,
            runner_type=cfg.runner_type,
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
            vocab_size=cfg.vocab_size,
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
    """Dispatch on `cfg.model_type` to the registered builder."""
    adapter = MODEL_ADAPTERS.get(cfg.model_type)
    if adapter is None:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
    return adapter.build(cfg, use_tp)


# ── Mesh / device ─────────────────────────────────────────────────────────────


def build_mesh(device_config: DeviceConfig) -> ttml.Mesh:
    """Construct a named mesh from `device_config`; axis names follow the C++ DP→TP order."""
    shape = tuple(int(s) for s in device_config.mesh_shape)
    n = len(shape)
    nontrivial = [i for i, s in enumerate(shape) if s > 1]
    is_line = len(nontrivial) <= 1
    enabled = (
        ("dp" if device_config.enable_ddp else None),
        ("tp" if device_config.enable_tp else None),
    )
    enabled_names = tuple(name for name in enabled if name is not None)

    # ttml.Mesh requires a name per axis; "_i" is a placeholder for axes
    # not assigned to DP or TP.
    axis_names = [f"_{i}" for i in range(n)]
    if not enabled_names:
        return ttml.Mesh(shape, tuple(axis_names))

    if is_line:
        if len(enabled_names) != 1:
            raise ValueError(
                f"Line mesh {shape} requires exactly one of enable_ddp / enable_tp; got enabled={enabled_names}"
            )
        active = nontrivial[0] if nontrivial else 0
        axis_names[active] = enabled_names[0]
    else:
        if len(enabled_names) != n:
            raise ValueError(f"2D mesh {shape} requires both axes assigned (DP and TP). Got enabled={enabled_names}")
        for i, name in enumerate(enabled_names):
            axis_names[i] = name
    return ttml.Mesh(shape, tuple(axis_names))


def _device_arch_name() -> str:
    """Short identifier for the active device's architecture (used in the header table)."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    if is_wormhole_b0(device):
        return "wormhole_b0"
    if is_blackhole(device):
        return "blackhole"
    return "unknown"


_HEADER_WIDTH = 70
# Content wraps within a 2-space margin on each side of the table.
_CONTENT_WIDTH = _HEADER_WIDTH - 4


def _print_banner(title: str) -> None:
    print("═" * _HEADER_WIDTH)
    print(f"  {title}")
    print("═" * _HEADER_WIDTH)


# Break a long value after whitespace or a path-like separator (/ . _ -), keeping the
# separator on the preceding line so paths and config names wrap at natural boundaries.
_WRAP_PIECE = re.compile(r"[^\s/._-]*[\s/._-]+|[^\s/._-]+")


def _wrap(text: str, width: int) -> list[str]:
    """Wrap `text` to `width`, preferring separator boundaries; hard-breaks tokens too long to fit. Always ≥1 line."""
    if len(text) <= width:
        return [text]
    lines, cur = [], ""
    for piece in _WRAP_PIECE.findall(text):
        if cur and len(cur) + len(piece) > width:
            lines.append(cur)
            cur = piece
        else:
            cur += piece
        while len(cur) > width:  # oversized spaceless token: hard-break at the edge
            lines.append(cur[:width])
            cur = cur[width:]
    if cur:
        lines.append(cur)
    return [line.rstrip() for line in lines]


def _print_section(name: str, fields: list[tuple[str, str]] | str, label_width: int | None = None) -> None:
    """Emit `── NAME ──…─` divider, then 2-space-indented `key  value` rows (or bare lines if `fields` is a string).

    Content wraps within a 2-space margin on both sides: string fields wrap back to the left indent,
    kv values wrap back to the column where the value starts. When `label_width` is provided, the key
    column is right-justified to that width — used to align value columns across multiple sections.
    """
    prefix = f"── {name.upper()} "
    print()
    print(prefix + "─" * (_HEADER_WIDTH - len(prefix)))
    if isinstance(fields, str):
        for line in fields.split("\n"):
            for chunk in _wrap(line, _CONTENT_WIDTH):
                print(f"  {chunk}")
        return
    if not fields:
        return
    width = label_width if label_width is not None else max(len(k) for k, _ in fields)
    value_width = max(1, _CONTENT_WIDTH - width - 2)  # minus label column + 2-space gap
    for label, value in fields:
        chunks = _wrap(value, value_width)
        print(f"  {label.rjust(width)}  {chunks[0]}")
        for chunk in chunks[1:]:
            print(f"  {' ' * width}  {chunk}")


def _print_header_close() -> None:
    print()
    print("═" * _HEADER_WIDTH)
    print()


def print_header(title: str, sections: list[tuple[str, list[tuple[str, str]] | str]]) -> None:
    """Print the full opening banner: title, then each section sharing one label width so values align vertically."""
    kv_widths = [max(len(k) for k, _ in fields) for _, fields in sections if isinstance(fields, list) and fields]
    width = max(kv_widths) if kv_widths else 0
    _print_banner(title)
    for name, fields in sections:
        _print_section(name, fields, label_width=width)
    _print_header_close()


def print_footer(title: str, fields: list[tuple[str, str]]) -> None:
    """Closing banner mirroring `_print_banner`: title rule, aligned `key  value` rows, closing rule."""
    width = max((len(k) for k, _ in fields), default=0)
    print()
    print("═" * _HEADER_WIDTH)
    print(f"  {title}")
    for label, value in fields:
        print(f"  {label.rjust(width)}  {value}")
    print("═" * _HEADER_WIDTH)
    print()


def get_device_peak_tflops_bf16() -> float:
    """Per-device theoretical BF16 TFLOPS. Whole-mesh peak = this × num_devices."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    # Per-core BF16 TFLOPS for each supported TT architecture.
    if is_wormhole_b0(device):
        per_core = 1.0
    elif is_blackhole(device):
        per_core = 1.35
    else:
        raise ValueError(f"Unknown device: {device.arch()}")
    return num_cores * per_core


# ── Dataset ───────────────────────────────────────────────────────────────────


class CausalLMDataset:
    """Sliding-window dataset of `{input_ids, labels}` samples; stride 1, labels shifted by one."""

    def __init__(self, tokens: np.ndarray, seq_length: int) -> None:
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self) -> int:
        if len(self.tokens) <= self.seq_length:
            return 0
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {
            "input_ids": self.tokens[index : index + self.seq_length].tolist(),
            "labels": self.tokens[index + 1 : index + self.seq_length + 1].tolist(),
        }


def build_dataset(data_path: str, seq_len: int, vocab_size: int) -> tuple[CausalLMDataset, CharTokenizer | None]:
    """Build (dataset, tokenizer). `.yaml`/`.yml` paths = pre-tokenized; otherwise plain text + char tokenizer."""
    is_pretokenized = data_path.endswith((".yaml", ".yml"))

    if is_pretokenized:
        if vocab_size == 0:
            raise ValueError(f"Pre-tokenized data ({data_path}) requires vocab_size in the model config.")
        with open(data_path, "r") as f:
            token_data = yaml.safe_load(f)
        tokens = np.array(token_data["tokens"], dtype=np.uint32)
        data_vocab_size = int(token_data["tokenizer_vocab_size"])
        max_token_id = int(tokens.max())
        if max_token_id >= vocab_size:
            raise ValueError(
                f"Tokenized data contains token id {max_token_id} but model vocab_size is "
                f"{vocab_size} (data file reports tokenizer_vocab_size={data_vocab_size})."
            )
        return CausalLMDataset(tokens, seq_len), None

    if vocab_size > 0:
        raise ValueError(
            f"Plain text data ({data_path}) uses character tokenization, which auto-detects "
            f"vocab_size. Remove vocab_size from the model config (or set it to 0)."
        )
    text = Path(data_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    tokens = np.array(tokenizer.encode(text), dtype=np.uint32)
    return CausalLMDataset(tokens, seq_len), tokenizer


# ── LR schedule ───────────────────────────────────────────────────────────────

_WARMUP_LINEAR_WARMUP_FRACTION = 0.1
_WARMUP_LINEAR_MIN_LR_FRACTION = 0.01


def build_lr_schedule(training_cfg: TrainingConfig, optimizer: Any, max_steps: int) -> Callable[[int], float]:
    """Return a `step -> lr` callable. `warmup_linear` runs warmup → linear decay; anything else is constant LR."""
    base_lr = optimizer.get_lr()

    if training_cfg.scheduler_type == "warmup_linear":
        warmup_steps = int(max_steps * _WARMUP_LINEAR_WARMUP_FRACTION)
        sched = SpeedrunScheduler(
            SchedulerConfig(
                max_lr=base_lr,
                min_lr=base_lr * _WARMUP_LINEAR_MIN_LR_FRACTION,
                warmup_steps=warmup_steps,
                hold_steps=0,
                total_steps=max_steps,
            )
        )
        return sched.lr_at

    return lambda _step: base_lr


# ── Callbacks ─────────────────────────────────────────────────────────────────


class DDPCallback(TrainerCallback):
    """All-reduce gradients across the `dp` mesh axis before each optimizer step."""

    def on_before_optimizer_step(self, trainer: SFTTrainer) -> None:
        ttml.sync_gradients(trainer.model.parameters())


class ThroughputCallback(TrainerCallback):
    """Print wall-clock TPS / TFLOPS / MFU every `log_interval` steps."""

    def __init__(self, flops_per_token: float, peak_tflops: float, log_interval: int = 1) -> None:
        self._flops_per_token = flops_per_token
        self._peak_tflops = peak_tflops
        self._log_interval = max(1, int(log_interval))
        self._step_start: float | None = None
        self._tokens_in_step: int = 0
        self._dp_size: int = 1

    def on_train_begin(self, trainer: SFTTrainer) -> None:
        mesh = ttml.mesh()
        self._dp_size = mesh.axis_size("dp") if mesh.has_axis("dp") else 1
        self._step_start = time.time()
        self._tokens_in_step = 0

    def on_after_forward(self, trainer: SFTTrainer, batch: Batch, loss: float) -> None:
        shape = batch.input_ids.shape()
        # Per-rank micro-shard tokens × dp_size = global tokens processed this step.
        self._tokens_in_step += int(shape[0]) * int(shape[-1]) * self._dp_size

    def on_step_end(self, trainer: SFTTrainer, step: int, step_loss: float = 0.0, *args: Any, **kwargs: Any) -> None:
        if step % self._log_interval != 0 or self._step_start is None:
            self._step_start = time.time()
            self._tokens_in_step = 0
            return
        now = time.time()
        elapsed_ms = (now - self._step_start) * 1000.0
        tps = self._tokens_in_step / max(1e-6, elapsed_ms / 1000.0)
        line = f"Step: {step}, Loss: {step_loss:.6f}, Time: {elapsed_ms:.2f} ms, TPS: {tps:.0f}"
        if self._flops_per_token > 0 and elapsed_ms > 0:
            achieved = tps * self._flops_per_token / 1e12
            line += f", TFLOPS: {achieved:.3g}"
            if self._peak_tflops > 0:
                mfu = achieved / self._peak_tflops * 100.0
                line += f", MFU: {mfu:.3g}%"
        print(line)
        self._step_start = now
        self._tokens_in_step = 0


class MemoryTrackerCallback(TrainerCallback):
    """Capture FORWARD_PASS / BACKWARD_PASS / FIRST_ITERATION_COMPLETE snapshots over step 1, then deregister."""

    def __init__(self) -> None:
        self._guard: Any | None = None

    def on_train_begin(self, trainer: SFTTrainer) -> None:
        self._guard = MemoryUsageTracker.begin_capture()

    def on_after_forward(self, trainer: SFTTrainer, batch: Batch, loss: float) -> None:
        MemoryUsageTracker.snapshot("FORWARD_PASS")

    def on_after_backward(self, trainer: SFTTrainer, batch: Batch) -> None:
        MemoryUsageTracker.snapshot("BACKWARD_PASS")

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
        MemoryUsageTracker.print_memory_usage()
        MemoryUsageTracker.clear()
        if self._guard is not None:
            self._guard.release()
            self._guard = None
        trainer.remove_callback(self)


class MoECallback(TrainerCallback):
    """DeepSeek-only: update expert routing bias each step; optionally log per-expert activation probs to CSV."""

    def __init__(self, log_path: str | None = None) -> None:
        self._log_path = log_path

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        if not hasattr(trainer.model, "get_moe_layers"):
            return
        moe_layers = trainer.model.get_moe_layers()
        # Log BEFORE update_expert_bias: update resets the underlying _token_counts buffer,
        # which the logger reads to compute activation probabilities.
        if self._log_path and moe_activation_logger.should_log_step(step):
            moe_activation_logger.log_step_expert_balance(self._log_path, step, moe_layers)
        for layer in moe_layers:
            layer.update_expert_bias()


class _AverageLossCallback(TrainerCallback):
    """Running mean of `step_loss` for the final-summary `Average loss` line."""

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def on_step_end(self, trainer: SFTTrainer, step: int, step_loss: float = 0.0, *args: Any, **kwargs: Any) -> None:
        self.sum += step_loss
        self.count += 1

    @property
    def average(self) -> float:
        return self.sum / max(1, self.count)


class ProfilerStepMarker(TrainerCallback):
    """Emit a per-step `iteration_<n>` marker, plus one-time `compilation_finished` after step 1."""

    def __init__(self) -> None:
        self._first: bool = True

    def on_step_end(self, trainer: SFTTrainer, step: int, *args: Any, **kwargs: Any) -> None:
        profiler_marker(None, f"iteration_{step}", dump_results=True)
        if self._first:
            profiler_marker(None, "compilation_finished")
            self._first = False


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def _serialize_params(model: Model) -> dict:
    """Serialize parameters into a pickle-friendly dict: float32 arrays + layout metadata per param."""
    state = {}
    for name, param in model.parameters().items():
        layout = param.tensor.get_value().get_layout()
        arr = param.tensor.to_numpy(ttnn.DataType.FLOAT32)
        state[name] = {
            "data": arr,
            "layout": layout.value if hasattr(layout, "value") else str(layout),
            "shape": arr.shape,
        }
    return state


def _restore_params(params: dict, model_state: dict) -> None:
    """Restore params from a `_serialize_params` dict into a parameter map (cast back to bf16)."""
    restored = set()
    for name, item in model_state.items():
        if name not in params:
            continue
        arr, layout_str = (item["data"], item.get("layout", "TILE")) if isinstance(item, dict) else (item, "TILE")
        layout = ttnn.Layout.ROW_MAJOR if "ROW_MAJOR" in str(layout_str) else ttnn.Layout.TILE
        tensor = ttml.autograd.Tensor.from_numpy(
            arr.astype(ml_dtypes.bfloat16), layout=layout, new_type=ttnn.DataType.BFLOAT16
        )
        params[name].assign(tensor)
        restored.add(name)

    missing = set(params) - restored  # in model, not restored → left at init (dangerous)
    unexpected = set(model_state) - set(params)  # in checkpoint, not in model → ignored
    if missing or unexpected:
        print(
            f"  [warn] checkpoint restore mismatch: "
            f"{len(missing)} model param(s) left at init "
            f"({sorted(missing)[:3]}{'…' if len(missing) > 3 else ''}), "
            f"{len(unexpected)} checkpoint param(s) ignored"
        )


def build_checkpoint_io(
    tokenizer: CharTokenizer | None,
    model_cfg: ModelConfig,
) -> tuple[Callable[[SFTTrainer, str], None], Callable[[SFTTrainer, str], int]]:
    """Return `(saver, loader)` closures. Saver writes step+params+tokenizer+model_config; loader restores params and returns step."""

    def saver(trainer: SFTTrainer, path: str) -> None:
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "step": trainer.step,
                    "model_state": _serialize_params(trainer.model),
                    "tokenizer": tokenizer,
                    "model_config": model_cfg,
                },
                f,
            )
        print(f"  Saved checkpoint to {path}")

    def loader(trainer: SFTTrainer, path: str) -> int:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        _restore_params(trainer.model.parameters(), ckpt["model_state"])
        return int(ckpt["step"])

    return saver, loader


def _shorten_home(path: str) -> str:
    """Replace the user's home directory prefix in `path` with `~` for display."""
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _prefixed(prefix: str, suffix: str) -> str:
    """Join with `_` when prefix is non-empty; bare `suffix` otherwise. Matches SFTTrainer's convention."""
    return f"{prefix}_{suffix}" if prefix else suffix


def find_latest_checkpoint(checkpoint_dir: str, checkpoint_prefix: str = "") -> str | None:
    """Highest-step `<prefix>_step_*.pkl` (or `step_*.pkl` if prefix empty) under `checkpoint_dir`. `<prefix>_final.pkl` wins ties."""
    directory = Path(checkpoint_dir)
    step_glob = _prefixed(checkpoint_prefix, "step_*.pkl")
    final_name = _prefixed(checkpoint_prefix, "final.pkl")
    files = list(directory.glob(step_glob))
    final_path = directory / final_name
    if final_path.exists():
        files.append(final_path)
    if not files:
        return None

    def step_of(p: Path) -> float:
        if p.name == final_name:
            return float("inf")
        m = re.search(r"step_(\d+)\.pkl$", p.name)
        return int(m.group(1)) if m else -1

    return str(max(files, key=step_of))


def _peek_checkpoint(path: str) -> tuple[int, ModelConfig]:
    """Read `(step, model_config)` from a checkpoint without restoring params or creating a model."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return int(ckpt["step"]), ckpt["model_config"]


def load_for_inference(
    path: str,
) -> tuple[Model, CharTokenizer, ModelConfig, int]:
    """Restore `(model, tokenizer, model_cfg, step)` from a checkpoint. Used to seed inference mode."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    model_cfg: ModelConfig = ckpt["model_config"]
    tokenizer: CharTokenizer = ckpt["tokenizer"]
    step = int(ckpt["step"])

    model = create_model(model_cfg, use_tp=False)
    _restore_params(model.parameters(), ckpt["model_state"])
    return model, tokenizer, model_cfg, step


# ── Inference ─────────────────────────────────────────────────────────────────

_MAX_RANDOM_SEED = 2**32 - 1


def _extract_last_step_logits(logits: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
    """Slice the final sequence position from `logits` regardless of tensor rank (handles 5D, 4D, and flat shapes)."""
    shape = logits.shape()
    if len(shape) == 5:
        seq = shape[3]
        sliced = ttnn.slice(
            logits.get_value(),
            [0, 0, 0, seq - 1, 0],
            [shape[0], shape[1], shape[2], seq, shape[4]],
        )
        return ttml.autograd.create_tensor(ttnn.reshape(sliced, [shape[0], 1, 1, shape[4]]), requires_grad=False)
    if len(shape) == 4:
        seq = shape[2]
        sliced = ttnn.slice(
            logits.get_value(),
            [0, 0, seq - 1, 0],
            [shape[0], shape[1], seq, shape[3]],
        )
        return ttml.autograd.create_tensor(ttnn.reshape(sliced, [shape[0], 1, 1, shape[3]]), requires_grad=False)
    # Fallback for any other rank: flatten to [positions, vocab] and keep the last position.
    # Assumes batch_size == 1 — with batch > 1 the batch dim folds into `positions`, so this
    # would keep only the last batch element's final position. Inference is single-prompt, so it holds.
    vocab = shape[-1]
    flat = ttnn.reshape(logits.get_value(), [-1, vocab])  # [positions, vocab]
    positions = flat.shape[0]
    if positions > 1:
        flat = ttnn.slice(flat, [positions - 1, 0], [positions, vocab])  # last position → [1, vocab]
    last = ttnn.reshape(flat, [1, 1, 1, vocab])  # canonicalize to [1, 1, 1, vocab]
    return ttml.autograd.create_tensor(last, requires_grad=False)


def _sample_next_token(
    last_logits: ttml.autograd.Tensor,
    vocab_size: int,
    temperature: float,
    top_k: int,
) -> int:
    """Pick the next token id. Argmax when `temperature ≈ 0`; otherwise top-k filter + temperature-scaled sample."""
    # Logits span the tile-padded vocab; columns >= the real vocab_size are padding slots with
    # arbitrary values. Bias them to -inf so neither argmax nor sampling can select an out-of-range
    # id — instead of clamping a bad id after the fact. last_logits is always [1, 1, 1, vocab].
    logits = last_logits.get_value()
    padded_vocab = logits.shape[3]
    if padded_vocab > vocab_size:
        bias_np = np.zeros((1, 1, 1, padded_vocab), dtype=np.float32)
        bias_np[..., vocab_size:] = -1e9
        bias = ttml.autograd.Tensor.from_numpy(
            bias_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
        ).get_value()
        last_logits = ttml.autograd.create_tensor(ttnn.add(logits, bias), requires_grad=False)

    if temperature < 0.01:
        argmax = ttnn.argmax(last_logits.get_value(), dim=3, keepdim=True)
        return int(argmax.item())
    if 0 < top_k < vocab_size:  # top_k >= vocab_size keeps everything → no filtering needed
        logits = last_logits.get_value()
        topk_values, topk_indices = ttnn.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        threshold = ttnn.slice(topk_values, [0, 0, 0, top_k - 1], [1, 1, 1, top_k])
        below_threshold = ttnn.lt(logits, threshold)
        neg_inf = ttnn.full_like(logits, -1e9, dtype=ttnn.bfloat16)
        filtered = ttnn.where(below_threshold, neg_inf, logits)
        for t in (topk_values, topk_indices, threshold, below_threshold, neg_inf):
            ttnn.deallocate(t)
        last_logits = ttml.autograd.create_tensor(filtered, requires_grad=False)
    seed = random.randint(0, _MAX_RANDOM_SEED)
    sampled = ttml.ops.sample.sample_op(last_logits, temperature, seed, None)
    return int(sampled.get_value().item())


def generate(
    model: Model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    sequence_length: int,
    mask: ttml.autograd.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> tuple[str, float, float]:
    """Generate up to `max_new_tokens` tokens from `prompt`; greedy when `temperature ≈ 0`, else `temperature`/`top_k` sampling.

    Does no printing. Returns `(text, elapsed_s, first_token_s)` — total generation time and the
    first token's time (compile-heavy), so callers can report total and steady-state speed.
    """
    model.eval()
    ttml.autograd.AutoContext.get_instance().reset_graph()
    device = ttml.autograd.AutoContext.get_instance().get_device()

    if not prompt:
        prompt = " "
    prompt_ids = tokenizer.encode(prompt)

    running = list(prompt_ids[:sequence_length])
    if len(running) < sequence_length:
        pad_id = tokenizer.stoi.get(" ", next(iter(tokenizer.stoi.values()), 0))
        running = [pad_id] * (sequence_length - len(running)) + running

    generated = []

    input_ttnn = ttnn.from_buffer(
        buffer=running[-sequence_length:],
        shape=[1, 1, 1, sequence_length],
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    vocab_size = tokenizer.vocab_size
    start = time.time()
    first_token_s = 0.0
    for step in range(max_new_tokens):
        model_input = ttml.autograd.create_tensor(input_ttnn, requires_grad=False)
        # Clone the mask each step; the autograd graph reset below invalidates prior tensors.
        mask_for_model = ttml.autograd.create_tensor(ttnn.clone(mask.get_value()), requires_grad=False)
        logits = model(model_input, mask_for_model)

        last_logits = _extract_last_step_logits(logits)
        next_id = _sample_next_token(last_logits, vocab_size, temperature, top_k)

        running.append(next_id)
        generated.append(next_id)

        shifted = ttnn.slice(input_ttnn, [0, 0, 0, 1], [1, 1, 1, sequence_length])
        new_token = ttnn.from_buffer(
            buffer=[next_id], shape=[1, 1, 1, 1], dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        input_ttnn = ttnn.concat([shifted, new_token], dim=3)
        ttnn.deallocate(shifted)
        ttnn.deallocate(new_token)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        if step == 0:
            first_token_s = time.time() - start

    elapsed = time.time() - start
    text = tokenizer.decode(generated)
    return text, elapsed, first_token_s


# ── Run modes ─────────────────────────────────────────────────────────────────


def _resolve_data_path(training_cfg: TrainingConfig) -> str:
    """Return explicit `training_cfg.data_path` if set; otherwise probe known shakespeare.txt locations."""
    if training_cfg.data_path:
        return training_cfg.data_path
    candidates = [
        Path("data/shakespeare.txt"),
        Path("tt-train/data/shakespeare.txt"),
        Path("../data/shakespeare.txt"),
        Path(__file__).resolve().parents[3] / "data" / "shakespeare.txt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No data_path set and shakespeare.txt not found in any of: {candidates}")


def run_training(
    args: argparse.Namespace,
    yaml_config: dict,
    training_cfg: TrainingConfig,
    model_cfg: ModelConfig,
    device_cfg: DeviceConfig,
) -> None:
    """Build dataset/model/optimizer/callbacks/trainer from configs + CLI overrides, then run training."""
    if args.track_memory:
        MemoryUsageTracker.snapshot("ENTRY")

    # Silent setup: gather everything, then emit a single header block before training.
    data_path = _resolve_data_path(training_cfg)
    seq_len = model_cfg.max_sequence_length
    dataset, tokenizer = build_dataset(data_path, seq_len, model_cfg.vocab_size)

    raw_vocab = tokenizer.vocab_size if tokenizer is not None else model_cfg.vocab_size
    model_cfg.vocab_size = round_up_to_tile(raw_vocab, 32)

    resume_path = None
    if not args.fresh:
        if args.resume:
            resume_path = args.resume
        elif args.checkpoint_dir:
            resume_path = find_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)

    resume_step: int | None = None
    if resume_path:
        try:
            resume_step, model_cfg = _peek_checkpoint(resume_path)
            seq_len = model_cfg.max_sequence_length
        except Exception:
            resume_path = None

    model = create_model(model_cfg, use_tp=device_cfg.enable_tp)
    if args.print_summary:
        summary(model)
    total_params = sum(math.prod(p.shape()) for p in model.parameters().values())

    flops_per_token = 0
    flops_fn = FLOPS_REGISTRY.get(model_cfg.model_type)
    if flops_fn is not None:
        flops_per_token = flops_fn(model.config, seq_len)

    if args.track_memory:
        MemoryUsageTracker.snapshot("MODEL_CREATION")

    optimizer = create_optimizer(model, yaml_config)
    if args.track_memory:
        MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")

    # DeepSeek's composite SDPA needs the mask passed explicitly; gpt2/llama/qwen3 use a fused
    # SDPA that materializes its own causal mask internally, so only build it for DeepSeek.
    attn_mask_for_model = None
    if model_cfg.model_type == "deepseek":
        attn_mask_for_model = ttml.autograd.Tensor.from_numpy(
            build_causal_mask(seq_len), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
        )

    mesh = ttml.mesh()
    mapper = mesh.axis_mapper("dp", tdim=0) if (mesh.has_axis("dp") and mesh.axis_size("dp") > 1) else None
    collate = partial(causal_lm_collate_fn, seq_len=seq_len, mapper=mapper)
    dataloader = InMemoryDataloader(dataset, collate, batch_size=training_cfg.batch_size, shuffle=True, drop_last=True)

    peak_tflops = get_device_peak_tflops_bf16() * mesh.num_devices() if flops_per_token > 0 else 0.0

    callbacks: list[TrainerCallback] = []

    # Training mechanics: mutate gradients / model state during the step.
    if mesh.has_axis("dp") and mesh.axis_size("dp") > 1:
        callbacks.append(DDPCallback())
    if model_cfg.model_type == "deepseek":
        callbacks.append(MoECallback(args.log_expert_activations))

    # Metrics.
    callbacks.append(ThroughputCallback(flops_per_token, peak_tflops, log_interval=1))
    avg_loss_cb = _AverageLossCallback()
    callbacks.append(avg_loss_cb)

    # Diagnostics / profiling (profiler last, so its per-step marker closes a full iteration).
    if args.track_memory:
        callbacks.append(MemoryTrackerCallback())
    callbacks.append(ProfilerStepMarker())

    if training_cfg.use_clip_grad_norm and device_cfg.enable_tp:
        raise ValueError("Clip grad norm is not supported with TP")

    saver, loader = build_checkpoint_io(tokenizer, model_cfg)

    # Match old train_nanogpt.py: training stops at min(max_steps, num_epochs × batches_per_epoch / grad_accum).
    batches_per_epoch = len(dataloader)
    grad_accum = max(1, training_cfg.gradient_accumulation_steps)
    epoch_bound = (training_cfg.num_epochs * batches_per_epoch) // grad_accum
    effective_max_steps = min(training_cfg.max_steps, epoch_bound) if epoch_bound > 0 else training_cfg.max_steps

    # Schedule over the steps actually run, so warmup + decay complete by the end of training.
    schedule = build_lr_schedule(training_cfg, optimizer, effective_max_steps)

    sft_cfg = SFTConfig(
        max_steps=effective_max_steps,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        save_interval=training_cfg.model_save_interval if args.checkpoint_dir else 0,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        max_grad_norm=(training_cfg.clip_grad_norm_max_norm if training_cfg.use_clip_grad_norm else 0.0),
        disable_progress_bar=True,
    )

    # TODO: investigate why we need fused cross_entropy_loss for training to work.
    def _causal_lm_loss(logits, batch):
        return ttml.ops.loss.cross_entropy_loss(logits, batch.labels, ttml.ops.ReduceType.MEAN)

    trainer = SFTTrainer(
        model=model,
        train_dataloader=dataloader,
        eval_dataloader=None,
        config=sft_cfg,
        optimizer=optimizer,
        lr_schedule=schedule,
        callbacks=callbacks,
        attention_mask=attn_mask_for_model,
        checkpoint_saver=saver,
        checkpoint_loader=loader,
        compute_loss_func=_causal_lm_loss,
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    # ─ Build header sections ─
    if raw_vocab and raw_vocab != model_cfg.vocab_size:
        vocab_str = f"{raw_vocab} chars → {model_cfg.vocab_size} padded"
    else:
        vocab_str = f"{model_cfg.vocab_size}"

    if training_cfg.scheduler_type == "warmup_linear":
        warmup = int(effective_max_steps * _WARMUP_LINEAR_WARMUP_FRACTION)
        schedule_str = f"warmup_linear · {warmup:,} warmup · {effective_max_steps - warmup:,} decay"
    else:
        schedule_str = "constant"

    model_fields: list[tuple[str, str]] = [
        (
            "arch",
            f"{model_cfg.num_blocks} layers · {model_cfg.embedding_dim} dim · {model_cfg.num_heads} heads · seq {seq_len}",
        ),
        ("params", f"{total_params:,}"),
    ]
    if flops_per_token > 0:
        model_fields.append(("flops", f"{flops_per_token / 1e9:.3g}G / token"))
    grad_ckpt = "on" if model_cfg.runner_type == ttml.models.RunnerType.MemoryEfficient else "off"
    model_fields.append(("grad ckpt", grad_ckpt))

    optimizer_fields: list[tuple[str, str]] = [
        ("name", optimizer.get_name()),
        ("lr", f"{optimizer.get_lr():.2e}"),
        ("schedule", schedule_str),
    ]
    if training_cfg.use_clip_grad_norm:
        optimizer_fields.append(("clip", f"max-norm {training_cfg.clip_grad_norm_max_norm}"))

    # Training section: run length, data coverage, and seed.
    steps_str = f"{effective_max_steps:,}"
    if effective_max_steps < training_cfg.max_steps:
        steps_str += f" (epoch-capped from {training_cfg.max_steps:,})"
    passes = effective_max_steps * grad_accum / batches_per_epoch if batches_per_epoch else 0.0
    training_fields: list[tuple[str, str]] = [
        ("steps", steps_str),
        ("epochs", f"{passes:.3g} ({training_cfg.num_epochs} configured)"),
        ("seed", str(training_cfg.seed)),
    ]

    dp_size = mesh.axis_size("dp") if mesh.has_axis("dp") else 1
    global_batch = training_cfg.batch_size * grad_accum * dp_size
    batch_str = (
        f"size {training_cfg.batch_size} · accum {training_cfg.gradient_accumulation_steps} "
        f"· global {global_batch:,} · dropout {model_cfg.dropout_prob}"
    )

    hardware_fields: list[tuple[str, str]] = [
        ("chip", _device_arch_name()),
        ("mesh", "×".join(str(s) for s in mesh.shape)),
    ]
    if peak_tflops > 0:
        hardware_fields.append(("peak", f"{peak_tflops:.1f} TFLOPS bf16"))
    hardware_fields.append(("memory", f"{get_available_device_memory_in_bytes() / (1024 * 1024):,.0f} MB"))

    ckpt_lines: list[str] = []
    if args.checkpoint_dir:
        ckpt_pattern = os.path.join(args.checkpoint_dir, _prefixed(args.checkpoint_prefix, "step_*.pkl"))
        ckpt_lines.append(f"save every {training_cfg.model_save_interval} → {_shorten_home(ckpt_pattern)}")
    if resume_path:
        ckpt_lines.append(f"resume {_shorten_home(resume_path)} @ step {resume_step}")

    diag_lines: list[str] = []
    if args.track_memory:
        diag_lines.append("memory tracking enabled")

    sections: list[tuple[str, list[tuple[str, str]] | str]] = [
        ("config", args.config),
        ("training", training_fields),
        (
            "data",
            [
                ("path", _shorten_home(data_path)),
                ("samples", f"{len(dataset):,}"),
                ("vocab", vocab_str),
            ],
        ),
        ("model", model_fields),
        ("optimizer", optimizer_fields),
        ("batch", batch_str),
        ("hardware", hardware_fields),
    ]
    if ckpt_lines:
        sections.append(("checkpoint", "\n".join(ckpt_lines)))
    if diag_lines:
        sections.append(("diagnostics", "\n".join(diag_lines)))

    print_header(f"{model_cfg.model_type} · training", sections)

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    if args.checkpoint_dir:
        final_path = os.path.join(args.checkpoint_dir, _prefixed(args.checkpoint_prefix, "final.pkl"))
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        saver(trainer, final_path)

    print()
    print("=" * 70)
    print("Training completed!")
    print(f"  - Total steps: {trainer.step}")
    print(f"  - Total time: {elapsed:.2f} s")
    print(f"  - Average loss: {avg_loss_cb.average:.6f}")
    print("=" * 70)


def run_inference(args: argparse.Namespace, seed: int) -> None:
    """Load checkpoint via `--model-path` and generate text from `--prompt`."""
    model, tokenizer, model_cfg, loaded_step = load_for_inference(args.model_path)
    seq_len = model_cfg.max_sequence_length

    causal_mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(seq_len), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    total_params = sum(math.prod(p.shape()) for p in model.parameters().values())
    vocab_str = (
        f"{tokenizer.vocab_size} → {model_cfg.vocab_size} padded"
        if tokenizer.vocab_size != model_cfg.vocab_size
        else str(model_cfg.vocab_size)
    )

    sections: list[tuple[str, list[tuple[str, str]] | str]] = [
        (
            "checkpoint",
            [
                ("path", _shorten_home(args.model_path)),
                ("step", f"{loaded_step:,}"),
            ],
        ),
        (
            "model",
            [
                (
                    "arch",
                    f"{model_cfg.num_blocks} layers · {model_cfg.embedding_dim} dim · {model_cfg.num_heads} heads · seq {seq_len}",
                ),
                ("params", f"{total_params:,}"),
                ("vocab", vocab_str),
            ],
        ),
        ("prompt", repr(args.prompt)),
        (
            "generate",
            [
                ("tokens", str(args.max_new_tokens)),
                ("temp", str(args.temperature)),
                ("top-k", str(args.top_k)),
                ("mode", "greedy" if args.temperature < 0.01 else "sampling"),
                ("seed", str(seed)),
            ],
        ),
        (
            "hardware",
            [
                ("chip", _device_arch_name()),
                ("mesh", "×".join(str(s) for s in ttml.mesh().shape)),
                ("memory", f"{get_available_device_memory_in_bytes() / (1024 * 1024):,.0f} MB"),
            ],
        ),
    ]
    print_header(f"{model_cfg.model_type} · inference", sections)

    text, elapsed, first_token_s = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        seq_len,
        causal_mask,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_divider = "── OUTPUT "
    print(output_divider + "─" * (_HEADER_WIDTH - len(output_divider)))
    print(text)

    n = args.max_new_tokens
    speed = f"{n / elapsed:.3g} tok/s" if elapsed > 0 else "n/a"
    if n > 1 and elapsed > first_token_s:
        speed += f"  ({(n - 1) / (elapsed - first_token_s):.3g} steady)"
    print_footer(
        f"{model_cfg.model_type} · generated",
        [("tokens", f"{n:,}"), ("time", f"{elapsed:.2f} s"), ("speed", speed)],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with grouped help + back-compat aliases for old train_nanogpt.py callers."""
    p = argparse.ArgumentParser(description="NanoGPT/Llama/DeepSeek/Qwen3 training (SFTTrainer)")

    g = p.add_argument_group("Config & data")
    g.add_argument(
        "-c",
        "--config",
        type=str,
        default="training_shakespeare_nanogpt_char.yaml",
        help="Path to training config YAML",
    )
    g.add_argument(
        "--data-path", dest="data_path", type=str, default="", help="Override training data path (default: from config)"
    )
    g.add_argument(
        "--sequence-length",
        dest="sequence_length",
        type=int,
        default=None,
        help="Override sequence length (default: from config)",
    )

    g = p.add_argument_group("Training overrides")
    g.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Override batch size")
    g.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="Override max training steps")
    g.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    g.add_argument(
        "--max-grad-norm",
        dest="max_grad_norm",
        type=float,
        default=None,
        help="Enable gradient clipping with this max-norm",
    )

    g = p.add_argument_group("Checkpointing")
    g.add_argument(
        "--checkpoint-dir", dest="checkpoint_dir", type=str, default="", help="Directory for saving/loading checkpoints"
    )
    g.add_argument(
        "--checkpoint-prefix",
        dest="checkpoint_prefix",
        type=str,
        default="nano_gpt",
        help="Prefix for checkpoint filenames",
    )
    g.add_argument("--resume", type=str, default="", help="Specific checkpoint to resume from (default: auto-detect)")
    g.add_argument("--fresh", action="store_true", help="Skip resume; train from scratch")

    g = p.add_argument_group("Inference")
    g.add_argument("--model-path", dest="model_path", type=str, default="", help="Checkpoint to load for inference")
    g.add_argument("--prompt", type=str, default="", help="Prompt text; with --model-path triggers inference mode")
    g.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=300, help="Tokens to generate")
    g.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy)")
    g.add_argument("--top-k", dest="top_k", type=int, default=40, help="Top-k sampling (0=disabled)")

    g = p.add_argument_group("Diagnostics")
    g.add_argument("--track-memory", dest="track_memory", action="store_true", help="Enable memory tracking callbacks")
    g.add_argument(
        "--print-summary", dest="print_summary", action="store_true", help="Print model layer-by-layer summary"
    )
    g.add_argument(
        "--log-expert-activations",
        dest="log_expert_activations",
        type=str,
        default=None,
        help="DeepSeek-only: CSV path for per-step expert activation probabilities",
    )

    # Silent underscore aliases for old train_nanogpt.py callers. Same dest as the
    # canonical hyphen flag; SUPPRESS hides them from --help.
    for flag, dest, type_name in (
        ("--data_path", "data_path", "str"),
        ("--sequence_length", "sequence_length", "int"),
        ("--batch_size", "batch_size", "int"),
        ("--max_steps", "max_steps", "int"),
        ("--checkpoint_dir", "checkpoint_dir", "str"),
        ("--checkpoint_prefix", "checkpoint_prefix", "str"),
        ("--max_new_tokens", "max_new_tokens", "int"),
        ("--top_k", "top_k", "int"),
        ("--model_path", "model_path", "str"),
    ):
        p.add_argument(
            flag, dest=dest, type={"str": str, "int": int}[type_name], default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
    p.add_argument(
        "--track_memory", dest="track_memory", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--print_summary", dest="print_summary", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    # Deprecated renames — mapped to the canonical name in _apply_backcompat with a stderr warning.
    p.add_argument("--num_epochs", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--clip_grad_norm", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--model_save_path", type=str, default="", help=argparse.SUPPRESS)

    return p


def _apply_backcompat(args: argparse.Namespace) -> None:
    """Translate deprecated flags into canonical (with stderr warnings) and validate inference-mode flag pairing."""
    if args.num_epochs is not None:
        print("warning: --num_epochs is deprecated; use --epochs", file=sys.stderr)
        if args.epochs is None:
            args.epochs = args.num_epochs
    if args.clip_grad_norm is not None:
        print("warning: --clip_grad_norm is deprecated; use --max-grad-norm", file=sys.stderr)
        if args.max_grad_norm is None:
            args.max_grad_norm = args.clip_grad_norm
    if args.model_save_path:
        if args.checkpoint_dir:
            raise SystemExit("error: --model_save_path and --checkpoint-dir are mutually exclusive")
        print(
            "warning: --model_save_path is deprecated; use --checkpoint-dir + --checkpoint-prefix",
            file=sys.stderr,
        )
        dirpart = os.path.dirname(args.model_save_path) or "."
        basepart = os.path.basename(args.model_save_path) or "nano_gpt"
        args.checkpoint_dir = dirpart
        args.checkpoint_prefix = basepart

    if bool(args.prompt) ^ bool(args.model_path):
        missing = "--model-path" if args.prompt else "--prompt"
        raise SystemExit(f"error: inference mode requires both --prompt and --model-path (missing: {missing})")


def parse_args() -> argparse.Namespace:
    args = _build_parser().parse_args()
    _apply_backcompat(args)
    return args


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, load configs + CLI overrides, open device mesh, dispatch to inference or training."""
    args = parse_args()

    tt_metal_root = get_tt_metal_runtime_root()
    tt_train_root = f"{tt_metal_root}/tt-train"
    configs_root = f"{tt_train_root}/configs"
    try:
        yaml_config = load_config(args.config, f"{configs_root}/training_configs")
        training_cfg = TrainingConfig(yaml_config)
        if training_cfg.model_config:
            model_yaml = load_config(training_cfg.model_config, tt_train_root)
            model_cfg = parse_model_config(model_yaml)
        else:
            print("Warning: no model_config specified; using defaults", file=sys.stderr)
            model_cfg = parse_model_config({})
    except FileNotFoundError as e:
        print(f"Error: config file not found: {e}", file=sys.stderr)
        raise

    if args.data_path:
        training_cfg.data_path = args.data_path
    if args.batch_size is not None:
        training_cfg.batch_size = args.batch_size
    if args.max_steps is not None:
        training_cfg.max_steps = args.max_steps
    if args.epochs is not None:
        training_cfg.num_epochs = args.epochs
    if args.max_grad_norm is not None:
        training_cfg.use_clip_grad_norm = True
        training_cfg.clip_grad_norm_max_norm = args.max_grad_norm
    if args.sequence_length is not None:
        model_cfg.max_sequence_length = args.sequence_length

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    device_cfg = DeviceConfig(yaml_config)

    # TP-sharded parameter tensors don't round-trip through our pickle format,
    # so reject checkpoint flags upfront under TP.
    if device_cfg.enable_tp:
        if args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is not supported with tensor parallelism")
        if args.resume:
            raise ValueError("--resume is not supported with tensor parallelism")

    mesh = build_mesh(device_cfg)
    ttml.open_device_mesh(mesh, tuple(device_cfg.device_ids) if device_cfg.device_ids else None)
    ttml.autograd.AutoContext.get_instance().get_device()
    ttml.autograd.AutoContext.get_instance().set_seed(training_cfg.seed)
    np.random.seed(training_cfg.seed)
    random.seed(training_cfg.seed)  # Python RNG drives the per-token sampling seed in inference

    inference_only = args.prompt and args.model_path
    try:
        if inference_only:
            run_inference(args, training_cfg.seed)
        else:
            run_training(args, yaml_config, training_cfg, model_cfg, device_cfg)
    finally:
        ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
