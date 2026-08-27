# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face product adaptor for Qwen3-32B."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.models.qwen3_32b.model import (
    DEFAULT_HF_REVISION,
    QWEN3_32B_ACCURACY,
    QWEN3_32B_PERFORMANCE,
    Qwen3_32B,
    Qwen3_32BPagedAttentionConfig,
    Qwen3_32BPrecisionConfig,
)

DEFAULT_HF_MODEL = "Qwen/Qwen3-32B"


def _local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


@dataclass(frozen=True)
class Qwen3_32BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Qwen3_32BRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int
    cluster_shape: list[int]
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = 32
    disable_batched_prefill: bool = False
    batched_prefill_batched_extract: bool = True

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        if num_cached_tokens != 0:
            return False
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
class Qwen3_32BForCausalLM:
    model: Qwen3_32B
    tokenizer: Any
    runtime_config: Qwen3_32BRuntimeConfig
    instruct: bool = True
    generation_config: Qwen3_32BGenerationConfig = field(default_factory=Qwen3_32BGenerationConfig)

    def __post_init__(self) -> None:
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Qwen3_32BGenerationConfig(
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
        trust_remote_code=True,
        local_files_only=_local_files_only(),
    )
    tokenizer.stop_tokens = list(_qwen_stop_token_ids(tokenizer))
    return tokenizer


def _qwen_stop_token_ids(tokenizer) -> tuple[int, ...]:
    ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.extend([eos] if isinstance(eos, int) else list(eos))
    for token in ("<|im_end|>", "<|im_start|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return tuple(dict.fromkeys(int(value) for value in ids))


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    if num_devices != 8:
        raise ValueError(f"Qwen3-32B supports exactly 8 devices (T3K), got {num_devices}")
    return tuple(length for length in (128, 1024) if length <= min(max_prefill_chunk_size, max_seq_len))


def _cache_path(hf_model: str, mesh_device, cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir)
    elif os.getenv("TT_CACHE_PATH"):
        path = Path(os.environ["TT_CACHE_PATH"])
    else:
        path = Path("model_cache") / hf_model / "T3K"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_checkpoint_config(hf_config, *, num_devices: int) -> None:
    expected = {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "head_dim": 128,
    }
    actual = {name: getattr(hf_config, name, None) for name in expected}
    mismatches = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
    if mismatches:
        raise ValueError(f"Unexpected Qwen3-32B geometry: {mismatches}")
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({hf_config.num_attention_heads}/{hf_config.num_key_value_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Qwen3-32B requires bias-free QKV projections")
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Qwen3-32B requires an untied LM head")


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Qwen3_32BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Qwen3_32BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Qwen3_32BForCausalLM:
    del dtype
    num_devices = mesh_device.get_num_devices()
    if num_devices != 8:
        raise ValueError(f"Qwen3-32B supports exactly 8 devices (T3K), got {num_devices}")
    ttnn.SetDefaultDevice(mesh_device)
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=_local_files_only(),
    )
    _validate_checkpoint_config(hf_config, num_devices=num_devices)
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    precision = (
        optimizations
        if isinstance(optimizations, Qwen3_32BPrecisionConfig)
        else (QWEN3_32B_PERFORMANCE if optimizations == "performance" else QWEN3_32B_ACCURACY)
    )
    if not isinstance(precision, Qwen3_32BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Qwen3_32BPrecisionConfig")
    if not (1 <= resolved_layers <= hf_config.num_hidden_layers):
        raise ValueError(f"n_layers must be in [1, {hf_config.num_hidden_layers}], got {resolved_layers}")
    cache_path = _cache_path(hf_model, mesh_device, cache_dir)
    if paged_attention_config is None:
        block_size = 32
    else:
        block_size = int(paged_attention_config.block_size)
    head_dim = int(getattr(hf_config, "head_dim", 0) or 0)
    if head_dim != 128:
        raise ValueError(f"Qwen3-32B must use explicit HF head_dim=128, got {head_dim}")
    model = Qwen3_32B.from_pretrained(
        mesh_device,
        hf_model,
        revision=hf_revision,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=resolved_layers,
        cache_dir=cache_path,
        precision=precision,
        block_size=block_size,
        executor_mode=True,
    )
    runtime = Qwen3_32BRuntimeConfig(
        model_name=hf_model,
        model_cache_path=cache_path,
        max_prefill_chunk_size=4096,
        max_context_len=int(getattr(hf_config, "max_position_embeddings", max_seq_len)),
        max_seq_len=max_seq_len,
        trace_prefill_supported_seq_lens=_trace_seq_lens(num_devices, 4096, max_seq_len),
        n_layers=resolved_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        cluster_shape=list(mesh_device.shape),
        kv_cache_dtype=precision.kv_cache_dtype,
        batched_prefill_batched_extract=not os.environ.get("DISABLE_BATCHED_EXTRACT"),
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    return Qwen3_32BForCausalLM(
        model=model,
        tokenizer=tokenizer,
        runtime_config=runtime,
        instruct=instruct,
    )
