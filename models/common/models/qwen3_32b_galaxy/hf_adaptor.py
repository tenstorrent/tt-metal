# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face product adaptor for the TTTv2 WH Galaxy Qwen3-32B path."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.models.galaxy.kv_contract import GalaxyPagedAttentionConfig
from models.common.models.galaxy.recipes import GALAXY_PHYSICAL_BATCH, validate_galaxy_mesh
from models.common.models.qwen3_32b_galaxy import weight_utils
from models.common.models.qwen3_32b_galaxy.model import (
    DEFAULT_HF_REVISION,
    QWEN3_32B_GALAXY_ACCURACY,
    QWEN3_32B_GALAXY_HF_MODEL,
    QWEN3_32B_GALAXY_PERFORMANCE,
    Qwen3_32BGalaxyLayerWeights,
    Qwen3_32BGalaxyModelParameters,
    Qwen3_32BGalaxyPrecision,
    Qwen3_32BGalaxyTransformer2D,
    Qwen3_32BGalaxyWeights,
    build_qwen3_32b_galaxy_model,
    default_paged_attention_config,
    parameters_from_hf_config,
)

DEFAULT_HF_MODEL = QWEN3_32B_GALAXY_HF_MODEL

#: Galaxy serves batched prefill as one physical batch of 32 rows with at least
#: 16 active rows, no cached prefixes, and a 2048-token ceiling.
GALAXY_BATCHED_PREFILL_PHYSICAL_BATCH = 32
GALAXY_BATCHED_PREFILL_MINIMUM_ACTIVE_ROWS = 16
GALAXY_BATCHED_PREFILL_MAX_SEQUENCE_LENGTH = 2048


def _local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


@dataclass(frozen=True)
class Qwen3_32BGalaxyGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Qwen3_32BGalaxyRuntimeConfig:
    """Immutable serving policy for the Galaxy Qwen reconstruction."""

    model_name: str
    model_cache_path: Path | None
    max_context_len: int
    max_seq_len: int
    max_prefill_chunk_size: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int = GALAXY_PHYSICAL_BATCH
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = GALAXY_BATCHED_PREFILL_PHYSICAL_BATCH
    minimum_active_prefill_rows: int = GALAXY_BATCHED_PREFILL_MINIMUM_ACTIVE_ROWS
    allow_cached_prefix_batching: bool = False
    batched_prefill_batched_extract: bool = True
    disable_batched_prefill: bool = False

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        if num_cached_tokens != 0:
            return False
        return (
            prefill_seq_len in self.trace_prefill_supported_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


def _chat_template_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def encode_prompt(tokenizer: Any, prompt_text: Any, system_prompt_text: Any = None, *, instruct: bool = True):
    if instruct:
        chat: Any = []
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
class Qwen3_32BGalaxyForCausalLM:
    """Product-facing handle over the Galaxy Qwen tensor model."""

    model: Qwen3_32BGalaxyTransformer2D
    tokenizer: Any
    runtime_config: Qwen3_32BGalaxyRuntimeConfig
    instruct: bool = True
    generation_config: Qwen3_32BGalaxyGenerationConfig = field(default_factory=Qwen3_32BGalaxyGenerationConfig)

    def __post_init__(self) -> None:
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Qwen3_32BGalaxyGenerationConfig(
                max_decode_tokens=self.generation_config.max_decode_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
                stop_token_ids=tuple(getattr(self.tokenizer, "stop_tokens", ()) or ()),
            )

    @property
    def model_name(self) -> str:
        return self.runtime_config.model_name

    @property
    def model_cache_path(self) -> Path | None:
        return self.runtime_config.model_cache_path

    @property
    def max_seq_len(self) -> int:
        return self.runtime_config.max_seq_len

    @property
    def max_context_len(self) -> int:
        return self.runtime_config.max_context_len

    def encode_prompt(self, prompt_text: Any, system_prompt_text: Any = None, instruct: bool | None = None):
        return encode_prompt(
            self.tokenizer,
            prompt_text,
            system_prompt_text,
            instruct=self.instruct if instruct is None else instruct,
        )

    def encode_chat(self, messages: Any):
        return self.encode_prompt(messages, instruct=True)

    def close(self) -> None:
        self.model.close()


def _qwen_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.extend([eos] if isinstance(eos, int) else list(eos))
    for token in ("<|im_end|>", "<|im_start|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return tuple(dict.fromkeys(int(value) for value in ids))


def load_tokenizer(hf_model: str = DEFAULT_HF_MODEL, hf_revision: str | None = DEFAULT_HF_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        revision=hf_revision,
        trust_remote_code=True,
        local_files_only=_local_files_only(),
    )
    tokenizer.stop_tokens = list(_qwen_stop_token_ids(tokenizer))
    return tokenizer


def _cache_path(hf_model: str, cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir)
    elif os.getenv("TT_CACHE_PATH"):
        path = Path(os.environ["TT_CACHE_PATH"])
    else:
        path = Path("model_cache") / hf_model / "galaxy_8x4"
    path.mkdir(parents=True, exist_ok=True)
    return path


def convert_hf_model_weights(hf: Any, *, params: Qwen3_32BGalaxyModelParameters) -> Qwen3_32BGalaxyWeights:
    """Convert every HF tensor the Galaxy graph builder consumes."""

    base = hf.model
    if params.n_layers > len(base.layers):
        raise ValueError(f"checkpoint has {len(base.layers)} layers, requested {params.n_layers}")
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, params.rope_table_len(), params.head_dim, torch.bfloat16
    )
    layers = []
    for index, layer in enumerate(base.layers[: params.n_layers]):
        wqkv, wo, q_norm, k_norm, wqkv_bias = weight_utils.attention_weights_from_hf_layer(layer.self_attn)
        if params.qk_norm and (q_norm is None or k_norm is None):
            raise ValueError(f"Qwen3-32B layer {index} is missing per-head Q/K normalization weights")
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        layers.append(
            Qwen3_32BGalaxyLayerWeights(
                wqkv=wqkv,
                wo=wo,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm),
                q_norm=q_norm,
                k_norm=k_norm,
                wqkv_bias=wqkv_bias,
            )
        )
    return Qwen3_32BGalaxyWeights(
        embedding=weight_utils.embedding_table_torch(base.embed_tokens),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        layers=tuple(layers),
        final_norm=weight_utils.rms_weight_torch(base.norm),
        lm_head=weight_utils.lm_head_weight_torch(
            hf.lm_head,
            dim=params.dim,
            vocab_size=params.vocab_size,
            padded_vocab_size=params.padded_vocab_size,
        ),
    )


def _resolve_precision(optimizations: str | Qwen3_32BGalaxyPrecision) -> Qwen3_32BGalaxyPrecision:
    if isinstance(optimizations, Qwen3_32BGalaxyPrecision):
        return optimizations
    if optimizations == "performance":
        return QWEN3_32B_GALAXY_PERFORMANCE
    if optimizations == "accuracy":
        return QWEN3_32B_GALAXY_ACCURACY
    raise TypeError("optimizations must be 'accuracy', 'performance', or a Qwen3_32BGalaxyPrecision")


def from_pretrained(
    mesh_device: Any,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = GALAXY_PHYSICAL_BATCH,
    max_seq_len: int = 2048,
    prefill_sequence_lengths: tuple[int, ...] = (128,),
    batched_prefill_sequence_lengths: tuple[int, ...] = (),
    chunked_prefill_sequence_lengths: tuple[int, ...] = (),
    optimizations: str | Qwen3_32BGalaxyPrecision = "accuracy",
    n_layers: int | None = None,
    paged_attention_config: GalaxyPagedAttentionConfig | None = None,
    enable_device_sampling: bool = True,
    # True, matching the production Galaxy model. The Galaxy attention decode
    # path selects the fused op on exactly the prefetcher condition -
    #     if self.use_prefetcher:
    #         q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(...)
    #     else:
    #         ... rotary_embedding_llama(q, ...); rotary_embedding_llama(k, ...)
    # - so on a prefetcher mesh the non-fused pair is the *fallback* path, kept
    # for Blackhole, and it expects a different cos/sin layout: `get_rot_mats`
    # returns [1, 1, local_batch, head_dim] for the non-fused decode op, while
    # `get_rm_rot_mats` expands to [1, expanded_batch, heads, head_dim] for the
    # fused one.
    #
    # Measured for Llama on `(8, 4)` with the non-fused pair (D-B25b): the decode
    # step wrote a K of |max| = inf into the cache at the current position while
    # V, which does not pass through RoPE, was exact at PCC 0.99973 - and the
    # prefix prefill wrote was still 0.99993. Qwen3-32B has 64 heads against 8 KV
    # heads, so the head-row asymmetry that exposes this is larger here, not
    # smaller.
    use_qk_fused_rotary: bool = True,
    cache_dir: Path | str | None = None,
    load_hf_model: Any = None,
) -> Qwen3_32BGalaxyForCausalLM:
    """Load `Qwen/Qwen3-32B` onto one WH Galaxy `(8, 4)` mesh.

    ``n_layers`` builds a layer subset, which is how the Milestone B one-layer
    numerical qualification runs against the real checkpoint.
    """

    validate_galaxy_mesh("Galaxy Qwen3-32B", mesh_device)
    ttnn.SetDefaultDevice(mesh_device)
    load_kwargs = {"revision": hf_revision, "local_files_only": _local_files_only()}
    hf_config = AutoConfig.from_pretrained(hf_model, **load_kwargs)
    params = parameters_from_hf_config(
        hf_config,
        n_layers=n_layers,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        prefill_sequence_lengths=prefill_sequence_lengths,
        batched_prefill_sequence_lengths=batched_prefill_sequence_lengths,
        chunked_prefill_sequence_lengths=chunked_prefill_sequence_lengths,
    )
    precision = _resolve_precision(optimizations)
    cache_path = _cache_path(hf_model, cache_dir)
    paged = paged_attention_config or default_paged_attention_config(params)

    # `load_hf_model` is a seam, not a convenience. The default loads all 62 GB
    # of a 64-layer checkpoint eagerly, once per process, which is right for the
    # accuracy gate and ruinous for anything that only needs a layer subset: the
    # three-runs-in-fresh-processes rule costs three full loads. Callers that want
    # a subset inject a loader that reads only the shards it needs - the tests use
    # `galaxy_checkpoint.load_layer_subset_causal_lm` - and this module stays
    # independent of the test tree rather than importing from it.
    hf = (
        load_hf_model()
        if load_hf_model is not None
        else AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16, **load_kwargs)
    )
    hf.eval()
    try:
        weights = convert_hf_model_weights(hf, params=params)
    finally:
        del hf
    model = build_qwen3_32b_galaxy_model(
        mesh_device,
        params=params,
        weights=weights,
        precision=precision,
        paged_attention_config=paged,
        enable_device_sampling=enable_device_sampling,
        use_qk_fused_rotary=use_qk_fused_rotary,
        cache_path=cache_path,
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    runtime_config = Qwen3_32BGalaxyRuntimeConfig(
        model_name=Path(hf_model).name,
        model_cache_path=cache_path,
        max_context_len=int(getattr(hf_config, "max_position_embeddings", max_seq_len)),
        max_seq_len=max_seq_len,
        max_prefill_chunk_size=GALAXY_BATCHED_PREFILL_MAX_SEQUENCE_LENGTH,
        trace_prefill_supported_seq_lens=params.prefill_sequence_lengths,
        n_layers=params.n_layers,
        n_kv_heads=params.n_kv_heads,
        head_dim=params.head_dim,
        max_batch_size=max_batch_size,
        kv_cache_dtype=precision.kv_cache_dtype,
        disable_batched_prefill=bool(os.getenv("DISABLE_BATCHED_PREFILL")),
    )
    return Qwen3_32BGalaxyForCausalLM(
        model=model,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        instruct=instruct,
        generation_config=Qwen3_32BGalaxyGenerationConfig(stop_token_ids=tuple(tokenizer.stop_tokens)),
    )


__all__ = [
    "DEFAULT_HF_MODEL",
    "DEFAULT_HF_REVISION",
    "GALAXY_BATCHED_PREFILL_MAX_SEQUENCE_LENGTH",
    "GALAXY_BATCHED_PREFILL_MINIMUM_ACTIVE_ROWS",
    "GALAXY_BATCHED_PREFILL_PHYSICAL_BATCH",
    "Qwen3_32BGalaxyForCausalLM",
    "Qwen3_32BGalaxyGenerationConfig",
    "Qwen3_32BGalaxyRuntimeConfig",
    "convert_hf_model_weights",
    "encode_prompt",
    "from_pretrained",
    "load_tokenizer",
]
