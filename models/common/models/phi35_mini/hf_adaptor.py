# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face provider boundary for ``microsoft/Phi-3.5-mini-instruct``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.models.phi35_mini import weight_utils
from models.common.models.phi35_mini.model import (
    PHI35_MINI_ACCURACY,
    PHI35_MINI_PERFORMANCE,
    Phi35MiniLayerWeights,
    Phi35MiniModelParameters,
    Phi35MiniPagedAttentionConfig,
    Phi35MiniPrecisionConfig,
    Phi35MiniTransformer,
    Phi35MiniWeights,
    build_phi35_mini_transformer_config,
)

DEFAULT_HF_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_HF_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"

# Longrope short factors cover positions below this; Hugging Face switches every position to the long
# factors once a sequence passes it. This port keeps the short-factor tables only.
MAX_SUPPORTED_SEQ_LEN = 4096


@dataclass(frozen=True)
class Phi35MiniGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Phi35MiniRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    supports_batched_prefill: bool = True
    # One batched prefill pass for a full 32-user wave: the planner runs a same-length bucket larger than this
    # one user at a time, not in several waves.
    max_prefill_batch_size: int = 32
    disable_batched_prefill: bool = False
    batched_prefill_batched_extract: bool = True

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        del num_cached_tokens
        return (
            prefill_seq_len in self.trace_prefill_supported_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


def _chat_template_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def encode_prompt(tokenizer, prompt_text, system_prompt_text=None, *, instruct=True):
    if instruct:
        messages = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                messages.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                messages.append({"role": "user", "content": prompt_text})
        else:
            messages = prompt_text
        try:
            return _chat_template_ids(
                tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            )
        except ValueError:
            pass
    return tokenizer.encode(prompt_text, add_special_tokens=False)


@dataclass
class Phi35MiniForCausalLM:
    model: Phi35MiniTransformer
    tokenizer: Any
    runtime_config: Phi35MiniRuntimeConfig
    instruct: bool = True
    generation_config: Phi35MiniGenerationConfig = field(default_factory=Phi35MiniGenerationConfig)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Phi35MiniGenerationConfig(
                max_decode_tokens=self.generation_config.max_decode_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
                stop_token_ids=tuple(getattr(self.tokenizer, "stop_tokens", ()) or ()),
            )

    @property
    def model_name(self):
        return self.runtime_config.model_name

    @property
    def model_cache_path(self):
        return self.runtime_config.model_cache_path

    @property
    def max_seq_len(self):
        return self.model.config.max_seq_len

    @property
    def max_context_len(self):
        return self.runtime_config.max_context_len

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=None):
        return encode_prompt(
            self.tokenizer,
            prompt_text,
            system_prompt_text,
            instruct=self.instruct if instruct is None else instruct,
        )

    def encode_chat(self, messages):
        return self.encode_prompt(messages, instruct=True)


def load_tokenizer(hf_model: str, hf_revision: str | None = DEFAULT_HF_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=os.getenv("CI") == "true",
    )
    eos = getattr(tokenizer, "eos_token_id", None)
    stops = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    end_of_turn = tokenizer.convert_tokens_to_ids("<|end|>")
    if isinstance(end_of_turn, int) and end_of_turn >= 0 and end_of_turn not in stops:
        stops.append(end_of_turn)
    tokenizer.stop_tokens = stops
    return tokenizer


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    if num_devices != 1:
        raise ValueError(f"Phi-3.5-mini supports single-device (P150) lanes only, got {num_devices} devices")
    step = int(os.environ.get("TT_PHI35_PREFILL_STEP", "0") or 0)
    lengths = tuple(range(step, 1025, step)) + (2048, 4096, 8192, 16384, 32768) if step else (128, 1024)
    return tuple(length for length in lengths if length <= min(max_prefill_chunk_size, max_seq_len))


def _cache_path(hf_model: str, mesh_device, cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir)
    elif os.getenv("TT_CACHE_PATH"):
        path = Path(os.environ["TT_CACHE_PATH"])
    else:
        path = Path("model_cache") / hf_model / "P150"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_checkpoint_config(hf_config) -> None:
    if hf_config.hidden_size % hf_config.num_attention_heads:
        raise ValueError("Phi-3.5-mini hidden_size must be divisible by num_attention_heads")
    rope_parameters = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
    if rope_type != "longrope":
        raise ValueError(f"Phi-3.5-mini requires longrope RoPE scaling, got {rope_type!r}")
    original_max = rope_parameters.get(
        "original_max_position_embeddings", getattr(hf_config, "original_max_position_embeddings", None)
    )
    if original_max != MAX_SUPPORTED_SEQ_LEN:
        raise ValueError(f"Phi-3.5-mini requires original_max_position_embeddings={MAX_SUPPORTED_SEQ_LEN}")
    partial_rotary_factor = getattr(hf_config, "partial_rotary_factor", None)
    if partial_rotary_factor is None:
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    if float(partial_rotary_factor) != 1.0:
        raise ValueError("Phi-3.5-mini requires full-head RoPE (partial_rotary_factor=1.0)")
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Phi-3.5-mini does not use QKV projection bias")


def convert_hf_model_weights(
    hf,
    hf_config,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> Phi35MiniWeights:
    """Extract fused Phi-3.5-mini projections and convert them to provider-neutral tensors."""

    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, rope_table_len, head_dim, torch.bfloat16
    )
    layers = []
    for layer in base.layers[:n_layers]:
        attention = layer.self_attn
        if any(getattr(attention, name, None) is not None for name in ("q_norm", "k_norm")):
            raise ValueError("Phi-3.5-mini does not use QK norm")
        if getattr(attention.qkv_proj, "bias", None) is not None:
            raise ValueError("Phi-3.5-mini does not use QKV projection bias")
        wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(attention, num_devices)
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        layers.append(
            Phi35MiniLayerWeights(
                wqkv=wqkv,
                wo=wo,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Phi-3.5-mini requires an untied LM head")
    return Phi35MiniWeights(
        embedding=weight_utils.embed_tokens_torch(base.embed_tokens),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        layers=tuple(layers),
        final_norm=weight_utils.rms_weight_torch(base.norm).to(torch.bfloat16),
        lm_head=hf.lm_head.weight.detach().to(torch.bfloat16).clone(),
    )


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Phi35MiniPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Phi35MiniPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Phi35MiniForCausalLM:
    del dtype
    num_devices = mesh_device.get_num_devices()
    if num_devices != 1:
        raise ValueError(f"Phi-3.5-mini supports single-device (P150) lanes only, got {num_devices} devices")
    if max_seq_len > MAX_SUPPORTED_SEQ_LEN:
        raise ValueError(
            f"max_seq_len={max_seq_len} exceeds {MAX_SUPPORTED_SEQ_LEN}: this port builds the longrope short-factor "
            "tables only"
        )
    ttnn.SetDefaultDevice(mesh_device)
    load_kwargs = dict(revision=hf_revision, local_files_only=os.getenv("CI") == "true")
    hf_config = AutoConfig.from_pretrained(hf_model, **load_kwargs)
    _validate_checkpoint_config(hf_config)
    hf = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16, **load_kwargs)
    hf.eval()
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    if (
        not isinstance(resolved_layers, int)
        or isinstance(resolved_layers, bool)
        or not 0 < resolved_layers <= hf_config.num_hidden_layers
    ):
        raise ValueError(f"n_layers must be in [1, {hf_config.num_hidden_layers}]")
    precision = (
        optimizations
        if isinstance(optimizations, Phi35MiniPrecisionConfig)
        else (PHI35_MINI_PERFORMANCE if optimizations == "performance" else PHI35_MINI_ACCURACY)
    )
    if not isinstance(precision, Phi35MiniPrecisionConfig) or (
        isinstance(optimizations, str) and optimizations not in ("accuracy", "performance")
    ):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Phi35MiniPrecisionConfig")
    cache_path = _cache_path(hf_model, mesh_device, cache_dir)
    if paged_attention_config is None:
        block_size = 32
        paged_attention_config = Phi35MiniPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    params = Phi35MiniModelParameters(
        dim=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        hidden_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    rope_table_len = MAX_SUPPORTED_SEQ_LEN
    weights = convert_hf_model_weights(
        hf,
        hf_config,
        n_layers=resolved_layers,
        num_devices=num_devices,
        rope_table_len=rope_table_len,
        head_dim=head_dim,
    )
    config = build_phi35_mini_transformer_config(
        mesh_device=mesh_device,
        params=params,
        weights=weights,
        n_layers=resolved_layers,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    model = Phi35MiniTransformer(config)
    max_prefill_chunk_size = 2048
    runtime_config = Phi35MiniRuntimeConfig(
        model_name=Path(hf_model).name,
        model_cache_path=cache_path,
        max_prefill_chunk_size=max_prefill_chunk_size,
        max_context_len=int(hf_config.max_position_embeddings),
        max_seq_len=max_seq_len,
        trace_prefill_supported_seq_lens=_trace_seq_lens(num_devices, max_prefill_chunk_size, max_seq_len),
        disable_batched_prefill=bool(os.getenv("DISABLE_BATCHED_PREFILL")),
        batched_prefill_batched_extract=not bool(os.getenv("DISABLE_BATCHED_EXTRACT")),
    )
    del hf
    return Phi35MiniForCausalLM(model=model, tokenizer=tokenizer, runtime_config=runtime_config, instruct=instruct)
