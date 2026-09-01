# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face product adaptor for Qwen2.5-72B-Instruct."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.models.qwen25_72b import weight_utils
from models.common.models.qwen25_72b.model import (
    DEFAULT_HF_REVISION,
    QWEN25_72B_ACCURACY,
    QWEN25_72B_PERFORMANCE,
    Qwen25_72B,
    Qwen25_72BConfig,
    Qwen25_72BLayerWeights,
    Qwen25_72BPagedAttentionConfig,
    Qwen25_72BPrecisionConfig,
    Qwen25_72BWeights,
    build_qwen25_72b_model,
)

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


@dataclass(frozen=True)
class Qwen25_72BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Qwen25_72BRuntimeConfig:
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
class Qwen25_72BForCausalLM:
    model: Qwen25_72B
    tokenizer: Any
    runtime_config: Qwen25_72BRuntimeConfig
    instruct: bool = True
    generation_config: Qwen25_72BGenerationConfig = field(default_factory=Qwen25_72BGenerationConfig)

    def __post_init__(self) -> None:
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Qwen25_72BGenerationConfig(
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
    stop_ids = _qwen_stop_token_ids(tokenizer)
    tokenizer.stop_tokens = list(stop_ids)
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
        raise ValueError(f"Qwen2.5-72B supports exactly 8 devices (T3K), got {num_devices}")
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
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 29568,
        "vocab_size": 152064,
    }
    actual = {name: getattr(hf_config, name, None) for name in expected}
    mismatches = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
    if mismatches:
        raise ValueError(f"Unexpected Qwen2.5-72B geometry: {mismatches}")
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({hf_config.num_attention_heads}/{hf_config.num_key_value_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Qwen2.5-72B requires an untied LM head")


def convert_hf_model_weights(
    hf,
    hf_config,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> Qwen25_72BWeights:
    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, rope_table_len, head_dim, torch.bfloat16
    )
    layers = []
    for layer in base.layers[:n_layers]:
        wqkv, wo, q_norm, k_norm, wqkv_bias = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_devices)
        if q_norm is not None or k_norm is not None:
            raise ValueError("Qwen2.5-72B does not support QK norm weights")
        if wqkv_bias is None:
            raise ValueError("Qwen2.5-72B requires QKV projection bias")
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        ff_align = 32 * 32 * num_devices
        ff_padded = math.ceil(w1.shape[-1] / ff_align) * ff_align
        if ff_padded != w1.shape[-1]:
            w1 = torch.nn.functional.pad(w1, (0, ff_padded - w1.shape[-1]))
            w3 = torch.nn.functional.pad(w3, (0, ff_padded - w3.shape[-1]))
            w2 = torch.nn.functional.pad(w2, (0, 0, 0, ff_padded - w2.shape[-2]))
        layers.append(
            Qwen25_72BLayerWeights(
                wqkv=wqkv,
                wo=wo,
                wqkv_bias=wqkv_bias.to(torch.bfloat16),
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )
    return Qwen25_72BWeights(
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
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Qwen25_72BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Qwen25_72BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Qwen25_72BForCausalLM:
    del dtype
    num_devices = mesh_device.get_num_devices()
    if num_devices != 8:
        raise ValueError(f"Qwen2.5-72B supports exactly 8 devices (T3K), got {num_devices}")
    ttnn.SetDefaultDevice(mesh_device)
    load_kwargs = {
        "revision": DEFAULT_HF_REVISION,
        "trust_remote_code": True,
        "local_files_only": _local_files_only(),
    }
    hf_config = AutoConfig.from_pretrained(hf_model, **load_kwargs)
    _validate_checkpoint_config(hf_config, num_devices=num_devices)
    hf = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16, **load_kwargs)
    hf.eval()
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    precision = (
        optimizations
        if isinstance(optimizations, Qwen25_72BPrecisionConfig)
        else (QWEN25_72B_PERFORMANCE if optimizations == "performance" else QWEN25_72B_ACCURACY)
    )
    if not isinstance(precision, Qwen25_72BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Qwen25_72BPrecisionConfig")
    cache_path = _cache_path(hf_model, mesh_device, cache_dir)
    block_size = 32
    if paged_attention_config is None:
        paged_attention_config = Qwen25_72BPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
    rope_table_len = ((max(max_seq_len * 2, 8192) + 127) // 128) * 128
    config = Qwen25_72BConfig(
        hf_model_id=hf_model,
        dim=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        hidden_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 1_000_000.0),
        num_hidden_layers=resolved_layers,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        rope_table_len=rope_table_len,
    )
    weights = convert_hf_model_weights(
        hf,
        hf_config,
        n_layers=resolved_layers,
        num_devices=num_devices,
        rope_table_len=rope_table_len,
        head_dim=head_dim,
    )
    model = build_qwen25_72b_model(
        mesh_device=mesh_device,
        config=config,
        weights=weights,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model, DEFAULT_HF_REVISION)
    stop_token_ids = _qwen_stop_token_ids(tokenizer)
    runtime_config = Qwen25_72BRuntimeConfig(
        model_name=Path(hf_model).name,
        model_cache_path=cache_path,
        max_prefill_chunk_size=2048,
        max_context_len=int(hf_config.max_position_embeddings),
        max_seq_len=max_seq_len,
        trace_prefill_supported_seq_lens=_trace_seq_lens(num_devices, 2048, max_seq_len),
        n_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_batch_size=max_batch_size,
        cluster_shape=list(mesh_device.shape),
        kv_cache_dtype=precision.kv_cache_dtype,
        disable_batched_prefill=bool(os.getenv("DISABLE_BATCHED_PREFILL")),
        batched_prefill_batched_extract=not bool(os.getenv("DISABLE_BATCHED_EXTRACT")),
    )
    del hf
    return Qwen25_72BForCausalLM(
        model=model,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        instruct=instruct,
        generation_config=Qwen25_72BGenerationConfig(stop_token_ids=stop_token_ids),
    )
