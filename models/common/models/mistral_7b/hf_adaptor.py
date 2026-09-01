# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face provider boundary for Mistral-7B-Instruct-v0.3."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.models.mistral_7b import weight_utils
from models.common.models.mistral_7b.model import (
    MISTRAL_ACCURACY,
    MISTRAL_PERFORMANCE,
    Mistral7B,
    Mistral7BLayerWeights,
    Mistral7BModelParameters,
    Mistral7BPagedAttentionConfig,
    Mistral7BPrecisionConfig,
    Mistral7BWeights,
    build_mistral_7b_transformer_config,
)

DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_HF_REVISION = None


@dataclass(frozen=True)
class Mistral7BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Mistral7BRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    supports_batched_prefill: bool = True
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
        chat = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
        else:
            chat = prompt_text
        try:
            return _chat_template_ids(tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
        except ValueError:
            pass
    return tokenizer.encode(prompt_text, add_special_tokens=False)


@dataclass
class Mistral7BForCausalLM:
    model: Mistral7B
    tokenizer: Any
    runtime_config: Mistral7BRuntimeConfig
    instruct: bool = True
    generation_config: Mistral7BGenerationConfig = field(default_factory=Mistral7BGenerationConfig)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            stops = tuple(getattr(self.tokenizer, "stop_tokens", ()) or ())
            self.generation_config = Mistral7BGenerationConfig(
                max_decode_tokens=self.generation_config.max_decode_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
                stop_token_ids=stops,
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
    tokenizer.stop_tokens = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    return tokenizer


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    allowed = {1: (128,), 2: (128, 1024), 8: (128, 1024)}.get(num_devices, (128,))
    return tuple(length for length in allowed if length <= min(max_prefill_chunk_size, max_seq_len))


def _cache_path(hf_model: str, mesh_device, cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir)
    elif os.getenv("TT_CACHE_PATH"):
        path = Path(os.environ["TT_CACHE_PATH"])
    else:
        topology = {1: "N150", 2: "N300", 8: "T3K"}.get(
            mesh_device.get_num_devices(), f"TP{mesh_device.get_num_devices()}"
        )
        path = Path("model_cache") / hf_model / topology
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_checkpoint_config(hf_config) -> None:
    if hf_config.hidden_size % hf_config.num_attention_heads:
        raise ValueError("Mistral hidden_size must be divisible by num_attention_heads")
    rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope_parameters.get("rope_theta", 1_000_000.0)
    rope_type = rope_parameters.get("rope_type", "default")
    if float(rope_theta) != 1_000_000.0 or rope_type != "default":
        raise ValueError("Mistral-7B-Instruct-v0.3 requires plain RoPE theta=1,000,000")
    if getattr(hf_config, "sliding_window", None) is not None:
        raise ValueError("Mistral-7B-Instruct-v0.3 requires full attention (sliding_window=None)")
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Mistral-7B-Instruct-v0.3 does not use QKV projection bias")


def convert_hf_model_weights(
    hf,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> Mistral7BWeights:
    """Extract and convert all Hugging Face tensors consumed by the TT builder."""

    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb,
        rope_table_len,
        head_dim,
        torch.bfloat16,
    )
    layers = []
    for layer in base.layers[:n_layers]:
        attention = layer.self_attn
        if any(getattr(attention, name, None) is not None for name in ("q_norm", "k_norm")):
            raise ValueError("Mistral-7B-Instruct-v0.3 does not use QK norm")
        if any(
            getattr(projection, "bias", None) is not None
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        ):
            raise ValueError("Mistral-7B-Instruct-v0.3 does not use QKV projection bias")
        wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(attention, num_devices)
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        layers.append(
            Mistral7BLayerWeights(
                wqkv=wqkv,
                wo=wo,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )
    return Mistral7BWeights(
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
    optimizations: str | Mistral7BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Mistral7BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Mistral7BForCausalLM:
    del dtype
    ttnn.SetDefaultDevice(mesh_device)
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=os.getenv("CI") == "true",
    )
    _validate_checkpoint_config(hf_config)
    num_devices = mesh_device.get_num_devices()
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({hf_config.num_attention_heads}/{hf_config.num_key_value_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    hf = AutoModelForCausalLM.from_pretrained(
        hf_model,
        revision=hf_revision,
        torch_dtype=torch.bfloat16,
        local_files_only=os.getenv("CI") == "true",
    )
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
        if isinstance(optimizations, Mistral7BPrecisionConfig)
        else (MISTRAL_PERFORMANCE if optimizations == "performance" else MISTRAL_ACCURACY)
    )
    if not isinstance(precision, Mistral7BPrecisionConfig) or (
        isinstance(optimizations, str) and optimizations not in ("accuracy", "performance")
    ):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Mistral7BPrecisionConfig")

    cache_path = _cache_path(hf_model, mesh_device, cache_dir)
    if paged_attention_config is None:
        block_size = 32
        paged_attention_config = Mistral7BPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    params = Mistral7BModelParameters(
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
    rope_table_len = math.ceil(max(max_seq_len * 2, 8192) / 128) * 128
    weights = convert_hf_model_weights(
        hf,
        n_layers=resolved_layers,
        num_devices=num_devices,
        rope_table_len=rope_table_len,
        head_dim=head_dim,
    )
    model_config = build_mistral_7b_transformer_config(
        mesh_device=mesh_device,
        params=params,
        weights=weights,
        n_layers=resolved_layers,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    model = Mistral7B(model_config)
    max_prefill_chunk_size = 2048
    runtime_config = Mistral7BRuntimeConfig(
        model_name=Path(hf_model).name,
        model_cache_path=cache_path,
        max_prefill_chunk_size=max_prefill_chunk_size,
        max_context_len=int(hf_config.max_position_embeddings),
        max_seq_len=max_seq_len,
        trace_prefill_supported_seq_lens=_trace_seq_lens(num_devices, max_prefill_chunk_size, max_seq_len),
        max_prefill_batch_size=8 if num_devices == 1 else 32,
        disable_batched_prefill=bool(os.getenv("DISABLE_BATCHED_PREFILL")),
        batched_prefill_batched_extract=not bool(os.getenv("DISABLE_BATCHED_EXTRACT")),
    )
    del hf
    return Mistral7BForCausalLM(
        model=model,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        instruct=instruct,
    )
