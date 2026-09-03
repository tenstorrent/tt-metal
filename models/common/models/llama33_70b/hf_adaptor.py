# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face product adaptor for the TTTv2 Llama 3.3 70B path."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.models.llama33_70b import weight_utils
from models.common.models.llama33_70b.model import (
    LLAMA33_70B_ACCURACY,
    LLAMA33_70B_PERFORMANCE,
    Llama33_70BLayerWeights,
    Llama33_70BModelParameters,
    Llama33_70BPagedAttentionConfig,
    Llama33_70BPrecisionConfig,
    Llama33_70BTransformer1D,
    Llama33_70BWeights,
    build_llama33_70b_transformer_1d_config,
)

DEFAULT_HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass(frozen=True)
class Llama33_70BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Llama33_70BRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    trace_prefill_warmup_seq_lens: tuple[int, ...] = ()
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
class Llama33_70BForCausalLM:
    model: Llama33_70BTransformer1D
    tokenizer: Any
    runtime_config: Llama33_70BRuntimeConfig
    instruct: bool = True
    generation_config: Llama33_70BGenerationConfig = field(default_factory=Llama33_70BGenerationConfig)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Llama33_70BGenerationConfig(
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


def load_tokenizer(hf_model: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model, local_files_only=os.getenv("CI") == "true")
    eos = getattr(tokenizer, "eos_token_id", None)
    tokenizer.stop_tokens = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    return tokenizer


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    if num_devices != 8:
        raise ValueError(f"Llama-3.3-70B supports exactly 8 devices (T3K), got {num_devices}")
    return tuple(length for length in (128, max_prefill_chunk_size) if length <= max_seq_len)


def _trace_warmup_seq_lens(max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    """Return logical representatives for regular and fixed-chunk traces."""

    candidates = (128, max_prefill_chunk_size, 2 * max_prefill_chunk_size)
    return tuple(dict.fromkeys(length for length in candidates if length <= max_seq_len))


def _cache_path(hf_model: str, mesh_device, cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir)
    elif os.getenv("TT_CACHE_PATH"):
        path = Path(os.environ["TT_CACHE_PATH"])
    else:
        path = Path("model_cache") / hf_model / "T3K"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stop_token_ids(hf) -> tuple[int, ...]:
    eos = getattr(getattr(hf, "generation_config", None), "eos_token_id", None)
    if eos is None:
        return ()
    values = (eos,) if isinstance(eos, int) else tuple(eos)
    return tuple(dict.fromkeys(int(value) for value in values))


def convert_hf_model_weights(
    hf,
    hf_config,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> Llama33_70BWeights:
    """Extract and convert every HF tensor consumed by the TT graph builder."""

    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, rope_table_len, head_dim, torch.bfloat16
    )
    layers = []
    for layer in base.layers[:n_layers]:
        wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_devices)
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        layers.append(
            Llama33_70BLayerWeights(
                wqkv=wqkv,
                wo=wo,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )
    return Llama33_70BWeights(
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
    optimizations: str | Llama33_70BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Llama33_70BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Llama33_70BForCausalLM:
    del dtype
    num_devices = mesh_device.get_num_devices()
    if num_devices != 8:
        raise ValueError(f"Llama-3.3-70B supports exactly 8 devices (T3K), got {num_devices}")
    ttnn.SetDefaultDevice(mesh_device)
    load_kwargs = {"local_files_only": os.getenv("CI") == "true"}
    hf_config = AutoConfig.from_pretrained(hf_model, **load_kwargs)
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({hf_config.num_attention_heads}/{hf_config.num_key_value_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    hf = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16, **load_kwargs)
    hf.eval()
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    precision = (
        optimizations
        if isinstance(optimizations, Llama33_70BPrecisionConfig)
        else (LLAMA33_70B_PERFORMANCE if optimizations == "performance" else LLAMA33_70B_ACCURACY)
    )
    if not isinstance(precision, Llama33_70BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Llama33_70BPrecisionConfig")
    cache_path = _cache_path(hf_model, mesh_device, cache_dir)
    if paged_attention_config is None:
        block_size = 32
        paged_attention_config = Llama33_70BPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
    params = Llama33_70BModelParameters(
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
    rope_table_len = ((max(max_seq_len * 2, 8192) + 127) // 128) * 128
    weights = convert_hf_model_weights(
        hf,
        hf_config,
        n_layers=resolved_layers,
        num_devices=num_devices,
        rope_table_len=rope_table_len,
        head_dim=head_dim,
    )
    model_config = build_llama33_70b_transformer_1d_config(
        mesh_device=mesh_device,
        params=params,
        weights=weights,
        n_layers=resolved_layers,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model)
    stop_token_ids = _stop_token_ids(hf)
    if stop_token_ids:
        tokenizer.stop_tokens = list(stop_token_ids)
    runtime_config = Llama33_70BRuntimeConfig(
        model_name=Path(hf_model).name,
        model_cache_path=cache_path,
        max_prefill_chunk_size=2048,
        max_context_len=int(hf_config.max_position_embeddings),
        max_seq_len=max_seq_len,
        trace_prefill_supported_seq_lens=_trace_seq_lens(num_devices, 2048, max_seq_len),
        trace_prefill_warmup_seq_lens=_trace_warmup_seq_lens(2048, max_seq_len),
        disable_batched_prefill=bool(os.getenv("DISABLE_BATCHED_PREFILL")),
    )
    del hf
    return Llama33_70BForCausalLM(
        model=Llama33_70BTransformer1D(model_config),
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        instruct=instruct,
        generation_config=Llama33_70BGenerationConfig(stop_token_ids=stop_token_ids),
    )
