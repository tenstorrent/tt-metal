# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import compressor_prefill
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, TensorMetadata
from models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke import (
    DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES,
    DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS,
    _prepare_decode_ffn_fanout,
    _run_ttnn_decode_decoder_layer_slice,
    build_torch_decode_attention_runtime_reference,
    build_torch_decode_cache_prep_reference,
    layer_decode_decoder_layer_selector_keys,
)
from models.demos.deepseek_v4_flash.real_decode_logits_smoke import (
    DEFAULT_DECODE_LOGITS_TOP_K,
    VocabMode,
    _dense_pcc_summary,
    _run_ttnn_decode_logits_from_hidden,
    _topk_accuracy_summary,
    _topk_summary,
    _validate_logits_args,
    build_torch_decode_logits_reference,
    load_decode_logits_weights,
)
from models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke import DEFAULT_DECODE_STACK_LAYERS
from models.demos.deepseek_v4_flash.real_ffn_smoke import (
    _accuracy_summary,
    _metadata_summary,
    _selection_summary,
    _tensor_summary,
)
from models.demos.deepseek_v4_flash.real_input_stack_logits_smoke import (
    DEFAULT_INPUT_STACK_MAX_BYTES,
    DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS,
    DEFAULT_INPUT_STACK_MAX_TENSORS,
    EmbeddingMode,
    InputEmbeddingPayload,
    load_input_embedding_payload,
)
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import decode_real_kv_projection_weights
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE,
    _query_projection_weights_only,
    _rotate_compressed_prefill_kv,
    _run_ttnn_attention_runtime_from_cache_boundary,
    build_torch_prefill_attention_runtime_reference,
    decode_real_prefill_attention_projection_weights,
    decode_real_prefill_runtime_sparse_weights_if_needed,
    layer_prefill_attention_runtime_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    DEFAULT_SEQUENCE_LENGTH,
    _run_ttnn_prefill_cache_prep,
)
from models.demos.deepseek_v4_flash.real_prefill_decoder_layer_smoke import (
    _load_missing_tensors,
    _metadata_groups,
    _prepare_prefill_ffn_fanout,
    _residual_add,
    _run_ttnn_prefill_decoder_layer_slice,
    _unique_keys,
    build_torch_prefill_decoder_layer_reference,
)

REAL_MULTI_TOKEN_DECODE_SMOKE_SCHEMA_VERSION = 1
DEFAULT_MULTI_TOKEN_DECODE_STEPS = 3
DEFAULT_MULTI_TOKEN_MAX_TENSORS = DEFAULT_INPUT_STACK_MAX_TENSORS + 3 * DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS
DEFAULT_MULTI_TOKEN_MAX_BYTES = DEFAULT_INPUT_STACK_MAX_BYTES + 3 * DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES
DecodeFeedMode = Literal["supplied"]


@dataclass
class _LayerAssets:
    layer: int
    q_weights: Any
    kv_weights: Any
    compressor_weights: Any
    indexer_weights: Any
    prefill_input_ids: torch.Tensor | None
    prefill_routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]] | None
    prefill_shared_weights: Mapping[str, torch.Tensor] | None
    prefill_reference: Mapping[str, Any] | None
    prefill_cache_reference: Mapping[str, torch.Tensor]
    prefill_activated_experts: Sequence[int]
    prefill_router_preview: Mapping[str, Any]
    attention_keys: Sequence[str]
    selector_keys: Sequence[str]
    ffn_keys: list[str]


@dataclass(frozen=True)
class _LayerCacheState:
    layer: int
    current_position: int
    attention_input_history: torch.Tensor
    sliding_window_cache: torch.Tensor
    compressed_kv: torch.Tensor
    index_compressed_kv: torch.Tensor


@dataclass(frozen=True)
class _LayerDecodeStep:
    layer: int
    decode_input_ids: torch.Tensor | None
    activated_experts: Sequence[int]
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]]
    shared_weights: Mapping[str, torch.Tensor]
    reference: Mapping[str, Any]
    router_preview: Mapping[str, Any]
    cache_before: _LayerCacheState
    cache_after: _LayerCacheState
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _DecodeStep:
    step_index: int
    current_position: int
    feed_token_id: int
    feed_input_ids: torch.Tensor
    initial_hidden: torch.Tensor
    stack_hidden: torch.Tensor
    layer_steps: tuple[_LayerDecodeStep, ...]


def run_real_multi_token_decode_smoke(
    snapshot_dir: str | Path,
    *,
    layers: Sequence[int] = DEFAULT_DECODE_STACK_LAYERS,
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    decode_steps: int = DEFAULT_MULTI_TOKEN_DECODE_STEPS,
    input_ids: Sequence[int] | torch.Tensor | None = None,
    input_id_start: int = 0,
    prompt_label: str | None = None,
    decode_feed_mode: DecodeFeedMode = "supplied",
    embedding_mode: EmbeddingMode = "slice",
    max_embedding_rows: int = DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS,
    vocab_mode: VocabMode = "slice",
    vocab_start: int = 0,
    vocab_size: int | None = None,
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K,
    max_tensors: int = DEFAULT_MULTI_TOKEN_MAX_TENSORS,
    max_bytes: int = DEFAULT_MULTI_TOKEN_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    prefill_hidden_pcc: float = 0.99,
    layer_hidden_pcc: float = 0.99,
    final_norm_pcc: float = 0.999,
    logits_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    residual_atol: float = 3e-1,
    logits_rtol: float = 1e-1,
    logits_atol: float = 1.0,
    top_logit_atol: float = 1.0,
) -> dict[str, Any]:
    """Run real input IDs through a carried-cache multi-token decode stack.

    This is deliberately still a stepping stone: decode input IDs are supplied
    deterministically by default, while each step reports the greedy top-1 IDs
    produced by the logits. That keeps sliced-vocab validation stable while
    proving the attention/compressed/indexer cache state advances across tokens.
    """

    wall_start = time.perf_counter()
    phase_start = wall_start
    layers = tuple(int(layer) for layer in layers)
    _validate_multi_token_args(
        layers=layers,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
        decode_feed_mode=decode_feed_mode,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        pcc_thresholds={
            "prefill_hidden_pcc": prefill_hidden_pcc,
            "layer_hidden_pcc": layer_hidden_pcc,
            "final_norm_pcc": final_norm_pcc,
            "logits_pcc": logits_pcc,
        },
    )
    _validate_logits_args(vocab_mode=vocab_mode, vocab_start=vocab_start, vocab_size=vocab_size, top_k=top_k)

    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    _validate_multi_token_runtime_config(
        config,
        layers=layers,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
    )
    input_source = "deterministic_contiguous_input_ids" if input_ids is None else "explicit_input_ids"
    token_ids = _normalize_multi_input_ids(
        input_ids,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
        vocab_size=config.vocab_size,
        input_id_start=input_id_start,
    )
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    embedding = load_input_embedding_payload(
        index,
        config=config,
        input_ids=token_ids,
        embedding_mode=embedding_mode,
        max_embedding_rows=max_embedding_rows,
    )
    remaining_tensors = max_tensors - 1
    remaining_bytes = max_bytes - embedding.metadata.nbytes
    if remaining_tensors <= 0:
        raise ValueError("max_tensors must leave room for the embedding payload")
    if remaining_bytes <= 0:
        raise ValueError("max_bytes must leave room for the embedding payload")
    setup_load_seconds = _elapsed_seconds(phase_start)

    tensors: dict[str, torch.Tensor] = {}
    metadata: list[TensorMetadata] = []
    initial_prefill_hidden = embedding.hidden_states[:, :, :prefill_seq_len, :].contiguous()
    decode_hidden_tokens = [
        embedding.hidden_states[:, :, prefill_seq_len + step : prefill_seq_len + step + 1, :].contiguous()
        for step in range(decode_steps)
    ]
    prefill_input_ids = token_ids[:, :prefill_seq_len].contiguous()
    supplied_decode_ids = token_ids[:, prefill_seq_len:].contiguous()

    phase_start = time.perf_counter()
    (
        layer_assets,
        initial_reference_caches,
        prefill_hidden,
        prefill_summaries,
        metadata,
    ) = _build_prefill_stack_reference(
        index,
        tensors,
        metadata,
        config=config,
        layers=layers,
        prefill_hidden=initial_prefill_hidden,
        prefill_input_ids=prefill_input_ids,
        max_tensors=remaining_tensors,
        max_bytes=remaining_bytes,
    )
    prefill_build_seconds = _elapsed_seconds(phase_start)

    reference_cache_states = initial_reference_caches
    decode_steps_reference: list[_DecodeStep] = []
    decode_build_step_seconds: list[float] = []
    for step_index in range(decode_steps):
        step_start = time.perf_counter()
        current_position = int(prefill_seq_len + step_index)
        feed_input_ids = supplied_decode_ids[:, step_index : step_index + 1].contiguous()
        decode_step, reference_cache_states, metadata = _build_reference_decode_step(
            index,
            tensors,
            metadata,
            layer_assets,
            reference_cache_states,
            config=config,
            step_index=step_index,
            current_position=current_position,
            initial_hidden=decode_hidden_tokens[step_index],
            feed_input_ids=feed_input_ids,
            max_tensors=remaining_tensors,
            max_bytes=remaining_bytes,
        )
        decode_steps_reference.append(decode_step)
        decode_build_step_seconds.append(_elapsed_seconds(step_start))

    phase_start = time.perf_counter()
    logits_weights = load_decode_logits_weights(
        index,
        config=config,
        vocab_mode=vocab_mode,
        vocab_start=vocab_start,
        vocab_size=vocab_size,
        already_loaded_metadata=metadata,
        max_tensors=remaining_tensors,
        max_bytes=remaining_bytes,
    )
    all_metadata = [*metadata, *logits_weights.metadata]
    logits_build_step_seconds: list[float] = []
    reference_logits_by_step = []
    for step in decode_steps_reference:
        step_start = time.perf_counter()
        reference_logits_by_step.append(
            build_torch_decode_logits_reference(
                step.stack_hidden,
                logits_weights.norm_weight,
                logits_weights.head_weight,
                config=config,
            )
        )
        logits_build_step_seconds.append(_elapsed_seconds(step_start))
    logits_build_seconds = _elapsed_seconds(phase_start)

    phase_start = time.perf_counter()
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layers=layers,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
        decode_feed_mode=decode_feed_mode,
        config=config,
        embedding=embedding,
        input_ids=token_ids,
        prompt_label=prompt_label,
        input_source=input_source,
        prefill_summaries=prefill_summaries,
        steps=decode_steps_reference,
        logits_references=reference_logits_by_step,
        metadata=all_metadata,
        layer_assets=layer_assets,
        logits_weights=logits_weights,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        top_k=top_k,
    )
    result["timing"] = {
        "setup_load_seconds": setup_load_seconds,
        "prefill_build_seconds": prefill_build_seconds,
        "decode_build_total_seconds": _sum_seconds(decode_build_step_seconds),
        "decode_build_step_seconds": [
            {
                "step_index": int(step.step_index),
                "current_position": int(step.current_position),
                "seconds": seconds,
            }
            for step, seconds in zip(decode_steps_reference, decode_build_step_seconds)
        ],
        "logits_build_total_seconds": logits_build_seconds,
        "logits_build_step_seconds": [
            {
                "step_index": int(step.step_index),
                "current_position": int(step.current_position),
                "seconds": seconds,
            }
            for step, seconds in zip(decode_steps_reference, logits_build_step_seconds)
        ],
        "result_build_seconds": _elapsed_seconds(phase_start),
        "ttnn": {},
        "end_to_end_wall_seconds": None,
    }

    if cpu_only:
        result.pop("_reference_prefill_tensors", None)
        result["mode"] = "cpu-reference"
        result["accuracy"] = {
            "cpu_reference": {
                "passed": True,
                "reason": "cpu-only requested; TTNN comparison was not run",
            }
        }
        result["passed"] = True
        result["timing"]["end_to_end_wall_seconds"] = _elapsed_seconds(wall_start)
        return result

    if prefill_seq_len % PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(
            f"TTNN multi-token smoke prefill_seq_len must be a multiple of "
            f"{PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE}, got {prefill_seq_len}"
        )

    phase_start = time.perf_counter()
    ttnn_outputs = _run_ttnn_multi_token_stack(
        tensors,
        layer_assets,
        decode_steps_reference,
        logits_weights,
        config=config,
        initial_prefill_hidden=initial_prefill_hidden,
        device_id=device_id,
    )
    result["timing"]["ttnn"] = ttnn_outputs.get("timing", {})
    result["timing"]["ttnn_total_seconds"] = _elapsed_seconds(phase_start)
    _augment_result_for_ttnn(
        result,
        ttnn_outputs=ttnn_outputs,
        reference_steps=decode_steps_reference,
        reference_logits_by_step=reference_logits_by_step,
        logits_weights=logits_weights,
        top_k=top_k,
        device_id=device_id,
        prefill_hidden_pcc=prefill_hidden_pcc,
        layer_hidden_pcc=layer_hidden_pcc,
        final_norm_pcc=final_norm_pcc,
        logits_pcc=logits_pcc,
        rtol=rtol,
        atol=atol,
        residual_atol=residual_atol,
        logits_rtol=logits_rtol,
        logits_atol=logits_atol,
        top_logit_atol=top_logit_atol,
    )
    result["timing"]["end_to_end_wall_seconds"] = _elapsed_seconds(wall_start)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real DeepSeek V4 Flash input IDs through a two-layer, carried-cache multi-token decode stack."
        )
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_DECODE_STACK_LAYERS))
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_MULTI_TOKEN_DECODE_STEPS)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--input-id-start", type=int, default=0)
    parser.add_argument("--prompt-label")
    parser.add_argument("--decode-feed-mode", choices=("supplied",), default="supplied")
    parser.add_argument("--embedding-mode", choices=("slice", "full"), default="slice")
    parser.add_argument("--max-embedding-rows", type=int, default=DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS)
    parser.add_argument(
        "--vocab-mode",
        choices=("full", "slice"),
        default=os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_VOCAB_MODE", "slice"),
    )
    parser.add_argument(
        "--vocab-start",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_VOCAB_START", 0),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_VOCAB_SIZE"),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MULTI_TOKEN_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MULTI_TOKEN_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--prefill-hidden-pcc", type=float, default=0.99)
    parser.add_argument("--layer-hidden-pcc", type=float, default=0.99)
    parser.add_argument("--final-norm-pcc", type=float, default=0.999)
    parser.add_argument("--logits-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--residual-atol", type=float, default=3e-1)
    parser.add_argument("--logits-rtol", type=float, default=1e-1)
    parser.add_argument("--logits-atol", type=float, default=1.0)
    parser.add_argument("--top-logit-atol", type=float, default=1.0)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_multi_token_decode_smoke(
        args.snapshot_dir,
        layers=args.layers,
        prefill_seq_len=args.prefill_seq_len,
        decode_steps=args.decode_steps,
        input_ids=args.input_ids,
        input_id_start=args.input_id_start,
        prompt_label=args.prompt_label,
        decode_feed_mode=args.decode_feed_mode,
        embedding_mode=args.embedding_mode,
        max_embedding_rows=args.max_embedding_rows,
        vocab_mode=args.vocab_mode,  # type: ignore[arg-type]
        vocab_start=args.vocab_start,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        prefill_hidden_pcc=args.prefill_hidden_pcc,
        layer_hidden_pcc=args.layer_hidden_pcc,
        final_norm_pcc=args.final_norm_pcc,
        logits_pcc=args.logits_pcc,
        rtol=args.rtol,
        atol=args.atol,
        residual_atol=args.residual_atol,
        logits_rtol=args.logits_rtol,
        logits_atol=args.logits_atol,
        top_logit_atol=args.top_logit_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _build_prefill_stack_reference(
    index: RealCheckpointTensorIndex,
    tensors: dict[str, torch.Tensor],
    metadata: list[TensorMetadata],
    *,
    config: DeepSeekV4FlashConfig,
    layers: Sequence[int],
    prefill_hidden: torch.Tensor,
    prefill_input_ids: torch.Tensor,
    max_tensors: int,
    max_bytes: int,
) -> tuple[list[_LayerAssets], tuple[_LayerCacheState, ...], torch.Tensor, list[dict[str, Any]], list[TensorMetadata]]:
    layer_assets: list[_LayerAssets] = []
    cache_states: list[_LayerCacheState] = []
    summaries: list[dict[str, Any]] = []
    current_hidden = prefill_hidden
    for layer_index, layer in enumerate(layers):
        materialize_prefill_output = layer_index != len(layers) - 1
        attention_keys = layer_prefill_attention_runtime_keys(index, config=config, layer=layer)
        selector_keys = layer_decode_decoder_layer_selector_keys(index, config=config, layer=layer)
        tensors, metadata = _load_missing_tensors(
            index,
            tensors,
            metadata,
            selector_keys,
            max_tensors=max_tensors,
            max_bytes=max_bytes,
        )
        q_weights = decode_real_prefill_attention_projection_weights(tensors, config=config, layer=layer)
        kv_weights = decode_real_kv_projection_weights(tensors, config=config, layer=layer)
        compressor_weights, indexer_weights = decode_real_prefill_runtime_sparse_weights_if_needed(
            tensors,
            config=config,
            layer=layer,
            seq_len=int(current_hidden.shape[2]),
        )
        prefill_cache_reference = build_torch_prefill_attention_runtime_reference(
            tensors,
            q_weights,
            kv_weights,
            compressor_weights,
            indexer_weights,
            config=config,
            layer=layer,
            activation=current_hidden,
            start_pos=0,
        )
        prefill_reference = None
        prefill_activated_experts: Sequence[int] = ()
        prefill_router_preview = _not_materialized_prefill_summary(layer=layer)
        prefill_routed_weights_by_expert = None
        prefill_shared_weights = None
        prefill_ffn_input_ids = None
        ffn_keys: list[str] = []
        if materialize_prefill_output:
            post_attention_residual = _residual_add(
                current_hidden,
                prefill_cache_reference["attention_output_projected"],
            )
            (
                tensors,
                metadata,
                ffn_keys,
                prefill_activated_experts,
                prefill_router_preview,
                prefill_routed_weights_by_expert,
                prefill_shared_weights,
                ffn_reference,
                prefill_ffn_input_ids,
            ) = _prepare_prefill_ffn_fanout(
                index,
                tensors,
                metadata,
                config=config,
                layer=layer,
                requested_expert=None,
                activation=post_attention_residual,
                max_tensors=max_tensors,
                max_bytes=max_bytes,
                input_ids=prefill_input_ids,
            )
            prefill_reference = build_torch_prefill_decoder_layer_reference(
                activation=current_hidden,
                attention_reference=prefill_cache_reference,
                ffn_reference=ffn_reference,
            )
            next_hidden = prefill_reference["post_ffn_residual"]
        else:
            next_hidden = current_hidden

        asset = _LayerAssets(
            layer=int(layer),
            q_weights=q_weights,
            kv_weights=kv_weights,
            compressor_weights=compressor_weights,
            indexer_weights=indexer_weights,
            prefill_input_ids=prefill_ffn_input_ids,
            prefill_routed_weights_by_expert=prefill_routed_weights_by_expert,
            prefill_shared_weights=prefill_shared_weights,
            prefill_reference=prefill_reference,
            prefill_cache_reference=prefill_cache_reference,
            prefill_activated_experts=prefill_activated_experts,
            prefill_router_preview=prefill_router_preview,
            attention_keys=attention_keys,
            selector_keys=selector_keys,
            ffn_keys=list(ffn_keys),
        )
        layer_assets.append(asset)
        cache_state = _initial_cache_state_from_prefill(layer=int(layer), reference=prefill_cache_reference)
        cache_states.append(cache_state)
        summaries.append(
            _prefill_layer_summary(
                asset,
                cache_state=cache_state,
                config=config,
                materialized_prefill_output=materialize_prefill_output,
            )
        )
        current_hidden = next_hidden
    return layer_assets, tuple(cache_states), current_hidden, summaries, metadata


def _build_reference_decode_step(
    index: RealCheckpointTensorIndex,
    tensors: dict[str, torch.Tensor],
    metadata: list[TensorMetadata],
    layer_assets: Sequence[_LayerAssets],
    cache_states: Sequence[_LayerCacheState],
    *,
    config: DeepSeekV4FlashConfig,
    step_index: int,
    current_position: int,
    initial_hidden: torch.Tensor,
    feed_input_ids: torch.Tensor,
    max_tensors: int,
    max_bytes: int,
) -> tuple[_DecodeStep, tuple[_LayerCacheState, ...], list[TensorMetadata]]:
    hidden = initial_hidden
    next_cache_states: list[_LayerCacheState] = []
    layer_steps: list[_LayerDecodeStep] = []
    for asset, cache_state in zip(layer_assets, cache_states):
        decode_cache_reference = build_torch_decode_cache_prep_reference(
            tensors,
            _query_projection_weights_only(asset.q_weights),
            asset.kv_weights,
            config=config,
            layer=asset.layer,
            activation=hidden,
            current_position=current_position,
        )
        attention_reference = build_torch_decode_attention_runtime_reference(
            _cache_state_mapping(cache_state),
            decode_cache_reference,
            asset.q_weights,
            tensors[f"layers.{asset.layer}.attn.attn_sink"],
            config=config,
            layer=asset.layer,
            current_position=current_position,
            indexer_weights=asset.indexer_weights,
        )
        post_attention_residual = _residual_add(hidden, attention_reference["attention_output_projected"])
        (
            tensors,
            metadata,
            decode_ffn_keys,
            activated_experts,
            router_preview,
            routed_weights_by_expert,
            shared_weights,
            ffn_reference,
            ffn_input_ids,
        ) = _prepare_decode_ffn_fanout(
            index,
            tensors,
            metadata,
            config=config,
            layer=asset.layer,
            requested_expert=None,
            activation=post_attention_residual,
            max_tensors=max_tensors,
            max_bytes=max_bytes,
            input_ids=feed_input_ids,
        )
        asset.ffn_keys = _unique_keys([*asset.ffn_keys, *decode_ffn_keys])
        decode_reference = {
            "decode_input_hidden_states": hidden,
            "decode_cache": decode_cache_reference,
            "attention": attention_reference,
            "post_attention_residual": post_attention_residual,
            "ffn": ffn_reference,
            "post_ffn_residual": ffn_reference["residual_output"],
        }
        next_cache_state = _advance_cache_state(
            cache_state,
            decode_cache=decode_cache_reference,
            attention=attention_reference,
            config=config,
            layer=asset.layer,
            compressor_weights=asset.compressor_weights,
            indexer_weights=asset.indexer_weights,
        )
        layer_step = _LayerDecodeStep(
            layer=asset.layer,
            decode_input_ids=ffn_input_ids,
            activated_experts=activated_experts,
            routed_weights_by_expert=routed_weights_by_expert,
            shared_weights=shared_weights,
            reference=decode_reference,
            router_preview=router_preview,
            cache_before=cache_state,
            cache_after=next_cache_state,
            summary=_decode_layer_summary(
                layer=asset.layer,
                config=config,
                current_position=current_position,
                cache_before=cache_state,
                cache_after=next_cache_state,
                reference=decode_reference,
                router_preview=router_preview,
                activated_experts=activated_experts,
                routed_weights_by_expert=routed_weights_by_expert,
            ),
        )
        layer_steps.append(layer_step)
        next_cache_states.append(next_cache_state)
        hidden = decode_reference["post_ffn_residual"]

    return (
        _DecodeStep(
            step_index=int(step_index),
            current_position=int(current_position),
            feed_token_id=int(feed_input_ids[0, 0].item()),
            feed_input_ids=feed_input_ids,
            initial_hidden=initial_hidden,
            stack_hidden=hidden,
            layer_steps=tuple(layer_steps),
        ),
        tuple(next_cache_states),
        metadata,
    )


def _run_ttnn_multi_token_stack(
    tensors: Mapping[str, torch.Tensor],
    layer_assets: Sequence[_LayerAssets],
    steps: Sequence[_DecodeStep],
    logits_weights,
    *,
    config: DeepSeekV4FlashConfig,
    initial_prefill_hidden: torch.Tensor,
    device_id: int,
) -> dict[str, Any]:
    ttnn_wall_start = time.perf_counter()
    prefill_hidden = initial_prefill_hidden
    cache_states: list[_LayerCacheState] = []
    prefill_hidden_by_layer: dict[int, torch.Tensor] = {}
    prefill_summaries: list[dict[str, Any]] = []
    prefill_layer_seconds: list[dict[str, Any]] = []
    prefill_start = time.perf_counter()
    for asset_index, asset in enumerate(layer_assets):
        layer_start = time.perf_counter()
        materialize_prefill_output = asset_index != len(layer_assets) - 1
        if materialize_prefill_output:
            if asset.prefill_routed_weights_by_expert is None or asset.prefill_shared_weights is None:
                raise ValueError(f"Layer {asset.layer} is missing prefill FFN weights for TTNN prefill")
            prefill_outputs = _run_ttnn_prefill_decoder_layer_slice(
                tensors,
                asset.q_weights,
                asset.kv_weights,
                asset.compressor_weights,
                asset.indexer_weights,
                asset.prefill_routed_weights_by_expert,
                asset.prefill_shared_weights,
                config=config,
                layer=asset.layer,
                input_ids=asset.prefill_input_ids,
                activation=prefill_hidden,
                start_pos=0,
                device_id=device_id,
            )
            prefill_hidden = prefill_outputs["post_ffn_residual"]
            prefill_hidden_by_layer[int(asset.layer)] = prefill_hidden
            cache_outputs = prefill_outputs
            prefill_summaries.append(
                {
                    "layer": int(asset.layer),
                    "post_ffn_residual": _tensor_summary(prefill_hidden),
                    "experts_executed": int(len(prefill_outputs["per_expert_routes"])),
                    "per_expert_routes": {
                        str(expert_id): _selection_summary(route)
                        for expert_id, route in prefill_outputs["per_expert_routes"].items()
                    },
                }
            )
        else:
            cache_outputs = _run_ttnn_prefill_cache_for_decode(
                tensors,
                asset,
                config=config,
                activation=prefill_hidden,
                device_id=device_id,
            )
            prefill_summaries.append(
                {
                    "layer": int(asset.layer),
                    "post_ffn_residual": None,
                    "experts_executed": 0,
                    "cache_only": True,
                }
            )
        cache_states.append(_initial_cache_state_from_prefill(layer=asset.layer, reference=cache_outputs))
        prefill_layer_seconds.append(
            {
                "layer": int(asset.layer),
                "cache_only": not materialize_prefill_output,
                "seconds": _elapsed_seconds(layer_start),
            }
        )
    prefill_total_seconds = _elapsed_seconds(prefill_start)

    step_outputs = []
    decode_step_seconds: list[dict[str, Any]] = []
    logits_step_seconds: list[dict[str, Any]] = []
    for step in steps:
        decode_start = time.perf_counter()
        hidden = step.initial_hidden
        next_cache_states = []
        layer_outputs = []
        for asset, cache_state, reference_layer_step in zip(layer_assets, cache_states, step.layer_steps):
            decode_outputs = _run_ttnn_decode_decoder_layer_slice(
                tensors,
                asset.q_weights,
                asset.kv_weights,
                reference_layer_step.routed_weights_by_expert,
                reference_layer_step.shared_weights,
                config=config,
                layer=asset.layer,
                input_ids=reference_layer_step.decode_input_ids,
                prefill_activation=cache_state.attention_input_history.unsqueeze(1),
                decode_activation=hidden,
                current_position=step.current_position,
                device_id=device_id,
                prefill_cache_outputs=_cache_state_mapping(cache_state),
                indexer_weights=asset.indexer_weights,
            )
            next_cache_state = _advance_cache_state(
                cache_state,
                decode_cache=decode_outputs["decode_cache"],
                attention=decode_outputs["attention"],
                config=config,
                layer=asset.layer,
                compressor_weights=asset.compressor_weights,
                indexer_weights=asset.indexer_weights,
            )
            hidden = decode_outputs["post_ffn_residual"]
            next_cache_states.append(next_cache_state)
            layer_outputs.append(
                {
                    "layer": int(asset.layer),
                    "decode_outputs": decode_outputs,
                    "cache_after": next_cache_state,
                }
            )
        decode_seconds = _elapsed_seconds(decode_start)
        logits_start = time.perf_counter()
        logits_outputs = _run_ttnn_decode_logits_from_hidden(
            hidden,
            logits_weights.norm_weight,
            logits_weights.head_weight,
            config=config,
            device_id=device_id,
        )
        logits_seconds = _elapsed_seconds(logits_start)
        decode_step_seconds.append(
            {
                "step_index": int(step.step_index),
                "current_position": int(step.current_position),
                "seconds": decode_seconds,
            }
        )
        logits_step_seconds.append(
            {
                "step_index": int(step.step_index),
                "current_position": int(step.current_position),
                "seconds": logits_seconds,
            }
        )
        step_outputs.append(
            {
                "step_index": int(step.step_index),
                "stack_hidden": hidden,
                "final_norm": logits_outputs["final_norm"],
                "logits": logits_outputs["logits"],
                "layers": layer_outputs,
            }
        )
        cache_states = next_cache_states
    return {
        "prefill_hidden_by_layer": prefill_hidden_by_layer,
        "prefill_summaries": prefill_summaries,
        "steps": step_outputs,
        "final_cache_states": tuple(cache_states),
        "timing": {
            "prefill_total_seconds": prefill_total_seconds,
            "prefill_layer_seconds": prefill_layer_seconds,
            "decode_total_seconds": _sum_seconds(item["seconds"] for item in decode_step_seconds),
            "decode_step_seconds": decode_step_seconds,
            "logits_total_seconds": _sum_seconds(item["seconds"] for item in logits_step_seconds),
            "logits_step_seconds": logits_step_seconds,
            "total_seconds": _elapsed_seconds(ttnn_wall_start),
        },
    }


def _run_ttnn_prefill_cache_for_decode(
    tensors: Mapping[str, torch.Tensor],
    asset: _LayerAssets,
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    device_id: int,
) -> dict[str, torch.Tensor]:
    cache_outputs = _run_ttnn_prefill_cache_prep(
        tensors,
        _query_projection_weights_only(asset.q_weights),
        asset.kv_weights,
        config=config,
        layer=asset.layer,
        activation=activation,
        start_pos=0,
        device_id=device_id,
    )
    if asset.compressor_weights is None:
        return cache_outputs
    sparse_outputs = _run_ttnn_attention_runtime_from_cache_boundary(
        cache_outputs,
        asset.q_weights,
        asset.compressor_weights,
        asset.indexer_weights,
        tensors[f"layers.{asset.layer}.attn.attn_sink"],
        config=config,
        layer=asset.layer,
        start_pos=0,
        device_id=device_id,
    )
    return {**cache_outputs, **sparse_outputs}


def _augment_result_for_ttnn(
    result: dict[str, Any],
    *,
    ttnn_outputs: Mapping[str, Any],
    reference_steps: Sequence[_DecodeStep],
    reference_logits_by_step: Sequence[Mapping[str, torch.Tensor]],
    logits_weights,
    top_k: int,
    device_id: int,
    prefill_hidden_pcc: float,
    layer_hidden_pcc: float,
    final_norm_pcc: float,
    logits_pcc: float,
    rtol: float,
    atol: float,
    residual_atol: float,
    logits_rtol: float,
    logits_atol: float,
    top_logit_atol: float,
) -> None:
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = _ttnn_ops_summary(result["layers"], result["vocab"]["mode"], int(result["decode_steps"]))
    result["ttnn"] = {
        "prefill": ttnn_outputs["prefill_summaries"],
        "final_cache": [_cache_summary(cache_state) for cache_state in ttnn_outputs["final_cache_states"]],
    }

    step_passes = []
    reference_top1_ids = []
    ttnn_top1_ids = []
    for step_result, reference_step, reference_logits, ttnn_step in zip(
        result["steps"],
        reference_steps,
        reference_logits_by_step,
        ttnn_outputs["steps"],
    ):
        ttnn_top_k = _topk_summary(
            ttnn_step["logits"],
            top_k=top_k,
            vocab_start=logits_weights.vocab_start,
        )
        reference_top_k = step_result["reference"]["top_k"]
        reference_top1_ids.append(int(reference_top_k[0]["id"]))
        ttnn_top1_ids.append(int(ttnn_top_k[0]["id"]))
        step_result["ttnn"] = {
            "stack_hidden": _tensor_summary(ttnn_step["stack_hidden"]),
            "final_norm": _tensor_summary(ttnn_step["final_norm"]),
            "logits": _tensor_summary(ttnn_step["logits"]),
            "top_k": ttnn_top_k,
            "layers": [_ttnn_layer_step_summary(layer_output) for layer_output in ttnn_step["layers"]],
        }
        top_k_accuracy = _topk_accuracy_summary(
            reference_logits["logits"],
            ttnn_step["logits"],
            top_k=top_k,
            vocab_start=logits_weights.vocab_start,
            top_logit_atol=top_logit_atol,
        )
        if (
            not top_k_accuracy["passed"]
            and top_k_accuracy["top1_match"]
            and top_k_accuracy["top_logit_max_abs"] <= top_logit_atol
        ):
            top_k_accuracy = {
                **top_k_accuracy,
                "passed": True,
                "pass_policy": "top1_match_with_top_logit_atol",
                "exact_requested_topk_set_required": False,
            }
        accuracy = {
            "stack_hidden": _accuracy_summary(
                reference_step.stack_hidden,
                ttnn_step["stack_hidden"],
                pcc_threshold=layer_hidden_pcc,
                rtol=rtol,
                atol=residual_atol,
            ),
            "final_norm": _accuracy_summary(
                reference_logits["final_norm"],
                ttnn_step["final_norm"],
                pcc_threshold=final_norm_pcc,
                rtol=rtol,
                atol=atol,
            ),
            "logits": _dense_pcc_summary(
                reference_logits["logits"],
                ttnn_step["logits"],
                pcc_threshold=logits_pcc,
                rtol=logits_rtol,
                atol=logits_atol,
            ),
            "top_k": top_k_accuracy,
        }
        for reference_layer_step, ttnn_layer_output in zip(reference_step.layer_steps, ttnn_step["layers"]):
            accuracy[f"layer_{reference_layer_step.layer}_decode_post_ffn_residual"] = _accuracy_summary(
                reference_layer_step.reference["post_ffn_residual"],
                ttnn_layer_output["decode_outputs"]["post_ffn_residual"],
                pcc_threshold=layer_hidden_pcc,
                rtol=rtol,
                atol=residual_atol,
            )
        step_result["accuracy"] = accuracy
        step_result["passed"] = all(item["passed"] for item in accuracy.values())
        step_passes.append(step_result["passed"])

    if reference_steps and int(result["layers"][0]) in ttnn_outputs["prefill_hidden_by_layer"]:
        first_layer = int(result["layers"][0])
        result["accuracy"] = {
            f"layer_{first_layer}_prefill_post_ffn_residual": _accuracy_summary(
                result["_reference_prefill_tensors"][f"layer_{first_layer}_post_ffn_residual"],
                ttnn_outputs["prefill_hidden_by_layer"][first_layer],
                pcc_threshold=prefill_hidden_pcc,
                rtol=rtol,
                atol=residual_atol,
            )
        }
    else:
        result["accuracy"] = {}
    result["generated"]["reference_top1_ids"] = reference_top1_ids
    result["generated"]["ttnn_top1_ids"] = ttnn_top1_ids
    result["generated"]["top1_ids_match"] = reference_top1_ids == ttnn_top1_ids
    result["passed"] = all(step_passes) and all(item["passed"] for item in result["accuracy"].values())
    result.pop("_reference_prefill_tensors", None)


def _base_result(
    *,
    snapshot_dir: Path,
    layers: Sequence[int],
    prefill_seq_len: int,
    decode_steps: int,
    decode_feed_mode: DecodeFeedMode,
    config: DeepSeekV4FlashConfig,
    embedding: InputEmbeddingPayload,
    input_ids: torch.Tensor,
    prompt_label: str | None,
    input_source: str,
    prefill_summaries: Sequence[Mapping[str, Any]],
    steps: Sequence[_DecodeStep],
    logits_references: Sequence[Mapping[str, torch.Tensor]],
    metadata: Sequence[TensorMetadata],
    layer_assets: Sequence[_LayerAssets],
    logits_weights,
    max_tensors: int,
    max_bytes: int,
    top_k: int,
) -> dict[str, Any]:
    payload_bytes = _payload_bytes(metadata, layer_assets=layer_assets, logits_weights=logits_weights)
    embedding_payload = {
        "embedding": int(embedding.metadata.nbytes),
        "total": int(embedding.metadata.nbytes),
    }
    steps_payload = [
        _step_result_summary(
            step,
            logits_reference=logits_reference,
            logits_weights=logits_weights,
            top_k=top_k,
        )
        for step, logits_reference in zip(steps, logits_references)
    ]
    reference_top1_ids = [int(step["reference"]["top_k"][0]["id"]) for step in steps_payload]
    return {
        "schema_version": REAL_MULTI_TOKEN_DECODE_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layers": [int(layer) for layer in layers],
        "prefill_sequence_length": int(prefill_seq_len),
        "decode_steps": int(decode_steps),
        "current_positions": [int(step.current_position) for step in steps],
        "next_position": int(prefill_seq_len + decode_steps),
        "activation_source": "real_input_ids_host_embedding_lookup",
        "decode_feed_mode": decode_feed_mode,
        "model": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "sliding_window": config.sliding_window,
            "num_experts_per_tok": config.num_experts_per_tok,
            "n_routed_experts": config.n_routed_experts,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "input": {
            "source": input_source,
            "prompt_label": prompt_label,
            "token_ids": [int(value) for value in input_ids[0].tolist()],
            "prefill_token_ids": [int(value) for value in input_ids[0, :prefill_seq_len].tolist()],
            "supplied_decode_token_ids": [int(value) for value in input_ids[0, prefill_seq_len:].tolist()],
            "prefill_tokens": int(prefill_seq_len),
            "decode_tokens": int(decode_steps),
        },
        "generated": {
            "feed_policy": (
                "deterministic supplied decode IDs are fed into embedding; generated top-1 IDs are reported only"
            ),
            "reference_top1_ids": reference_top1_ids,
            "ttnn_top1_ids": [],
            "top1_ids_match": None,
            "limitation": (
                "sliced-vocab default top-1 may not be the full-vocab next token, so this runner does not feed "
                "generated IDs by default"
            ),
        },
        "embedding": {
            "mode": embedding.mode,
            "loaded_key": "embed.weight",
            "loaded_source_key": embedding.metadata.source_key,
            "loaded_shape": list(embedding.metadata.shape),
            "full_shape": [embedding.full_vocab_size, embedding.hidden_size],
            "row_start": int(embedding.row_start),
            "row_end": int(embedding.row_end),
            "deterministic_slice": None
            if embedding.mode == "full"
            else f"[{embedding.row_start}, {embedding.row_end})",
            "payload_bytes": embedding_payload,
            "embedded_hidden_states": _tensor_summary(embedding.hidden_states),
        },
        "stack_scope": {
            "path": (
                "real input_ids -> host embedding lookup -> layer 2 prefill output and layer 3 prefill cache -> "
                "multi-token decode with carried per-layer attention/compressed/indexer cache state -> final norm -> LM head"
            ),
            "cache_carry": (
                "each decode step receives the previous step's host-visible sliding-window KV plus compressed/indexer "
                "cache state; the cache is advanced once per layer per token"
            ),
            "decode_ffn_full_expert_fanout": (
                "enabled for every router top-k expert selected by each materialized decode token"
            ),
            "prefill_ffn_full_expert_fanout": (
                "enabled for every router top-k expert selected by each materialized layer-2 prefill token"
            ),
            "serving_eval_boundaries": "tokenizer, vLLM serving, all-layer model eval, and optimized device cache are excluded",
        },
        "prefill": list(prefill_summaries),
        "steps": steps_payload,
        "loaded_tensors": [_metadata_summary(item) for item in [*metadata, embedding.metadata]],
        "loaded_tensor_groups": {
            **_loaded_tensor_groups(metadata, layer_assets, logits_weights=logits_weights),
            "embedding": {
                "count": 1,
                "payload_bytes": embedding_payload,
                "canonical_keys": [embedding.metadata.canonical_key],
            },
        },
        "payload_bytes": {
            **payload_bytes,
            "embedding": embedding_payload,
            "total": int(payload_bytes["total"]) + int(embedding.metadata.nbytes),
        },
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata) + 1,
            "selected_payload_bytes": int(payload_bytes["total"]) + int(embedding.metadata.nbytes),
        },
        "vocab": {
            "mode": logits_weights.vocab_mode,
            "is_sliced": logits_weights.vocab_mode == "slice",
            "vocab_start": logits_weights.vocab_start,
            "vocab_size": logits_weights.vocab_size,
            "full_vocab_size": logits_weights.full_vocab_size,
            "deterministic_slice": (
                None
                if logits_weights.vocab_mode == "full"
                else f"[{logits_weights.vocab_start}, {logits_weights.vocab_start + logits_weights.vocab_size})"
            ),
        },
        "final_norm_lm_head": {
            "loaded_keys": {
                "final_norm": "norm.weight",
                "lm_head": logits_weights.head_key,
            },
            "loaded_source_keys": {
                "final_norm": "norm.weight",
                "lm_head": logits_weights.head_source,
            },
            "payload_bytes": payload_bytes["final_norm_lm_head"],
            "lm_head_shape_loaded": list(logits_weights.head_weight.shape),
            "lm_head_full_shape": [logits_weights.full_vocab_size, int(config.hidden_size)],
        },
        "host_boundaries": _host_boundaries(logits_weights.vocab_mode),
        "reference_ops": _reference_ops(layers, logits_weights.vocab_mode, decode_steps),
        "ttnn_ops": [],
        "ttnn": {},
        "accuracy": {},
        "passed": False,
        "_reference_prefill_tensors": {
            f"layer_{asset.layer}_post_ffn_residual": asset.prefill_reference["post_ffn_residual"]
            for asset in layer_assets
            if asset.prefill_reference is not None
        },
    }


def _step_result_summary(
    step: _DecodeStep,
    *,
    logits_reference: Mapping[str, torch.Tensor],
    logits_weights,
    top_k: int,
) -> dict[str, Any]:
    return {
        "step_index": int(step.step_index),
        "current_position": int(step.current_position),
        "next_position": int(step.current_position + 1),
        "feed_token_id": int(step.feed_token_id),
        "layers": [dict(layer_step.summary) for layer_step in step.layer_steps],
        "output_shapes": {
            "initial_decode_hidden": list(step.initial_hidden.shape),
            "stack_hidden": list(step.stack_hidden.shape),
            "final_norm": list(logits_reference["final_norm"].shape),
            "logits": list(logits_reference["logits"].shape),
        },
        "reference": {
            "stack_hidden": _tensor_summary(step.stack_hidden),
            "final_norm": _tensor_summary(logits_reference["final_norm"]),
            "logits": _tensor_summary(logits_reference["logits"]),
            "top_k": _topk_summary(
                logits_reference["logits"],
                top_k=top_k,
                vocab_start=logits_weights.vocab_start,
            ),
        },
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _initial_cache_state_from_prefill(layer: int, reference: Mapping[str, torch.Tensor]) -> _LayerCacheState:
    attention_history = reference["attn_norm_output"][:, 0].contiguous()
    head_dim = int(reference["kv_cache_ready"].shape[-1])
    index_head_dim = int(reference.get("index_compressed_kv", attention_history.new_empty(1, 0, 0)).shape[-1])
    if index_head_dim == 0:
        index_head_dim = head_dim
    return _LayerCacheState(
        layer=int(layer),
        current_position=int(attention_history.shape[1]),
        attention_input_history=attention_history,
        sliding_window_cache=reference["sliding_window_cache"].contiguous(),
        compressed_kv=reference.get("compressed_kv", attention_history.new_empty(1, 0, head_dim)).contiguous(),
        index_compressed_kv=reference.get(
            "index_compressed_kv",
            attention_history.new_empty(1, 0, index_head_dim),
        ).contiguous(),
    )


def _advance_cache_state(
    cache_state: _LayerCacheState,
    *,
    decode_cache: Mapping[str, torch.Tensor],
    attention: Mapping[str, torch.Tensor],
    config: DeepSeekV4FlashConfig,
    layer: int,
    compressor_weights: Any,
    indexer_weights: Any,
) -> _LayerCacheState:
    current_attention_input = decode_cache["attn_norm_output"][:, 0].contiguous()
    next_history = torch.cat([cache_state.attention_input_history, current_attention_input], dim=1).contiguous()
    compressed_kv, index_compressed_kv = _compressed_caches_from_attention_history(
        next_history,
        config=config,
        layer=layer,
        compressor_weights=compressor_weights,
        indexer_weights=indexer_weights,
    )
    return _LayerCacheState(
        layer=int(layer),
        current_position=int(cache_state.current_position + 1),
        attention_input_history=next_history,
        sliding_window_cache=attention["sliding_window_cache"].contiguous(),
        compressed_kv=compressed_kv,
        index_compressed_kv=index_compressed_kv,
    )


def _compressed_caches_from_attention_history(
    attention_input_history: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    compressor_weights: Any,
    indexer_weights: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = attention_input_history.shape
    compress_ratio = int(config.compress_ratios[layer])
    if compressor_weights is None or compress_ratio <= 0 or seq_len < compress_ratio:
        return (
            attention_input_history.new_empty(batch_size, 0, int(config.head_dim)),
            attention_input_history.new_empty(batch_size, 0, int(config.index_head_dim)),
        )
    compressed_kv = compressor_prefill(
        attention_input_history.to(torch.bfloat16),
        compressor_weights.wkv,
        compressor_weights.wgate,
        compressor_weights.ape,
        compressor_weights.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=int(config.head_dim),
        norm_eps=float(config.rms_norm_eps),
    ).to(torch.bfloat16)
    compressed_kv = _rotate_compressed_prefill_kv(
        compressed_kv,
        config=config,
        layer=layer,
        seq_len=seq_len,
        start_pos=0,
        compress_ratio=compress_ratio,
    )
    if indexer_weights is None:
        return compressed_kv.contiguous(), attention_input_history.new_empty(
            batch_size,
            0,
            int(config.index_head_dim),
        )
    index_compressed_kv = compressor_prefill(
        attention_input_history.to(torch.bfloat16),
        indexer_weights.compressor.wkv,
        indexer_weights.compressor.wgate,
        indexer_weights.compressor.ape,
        indexer_weights.compressor.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=int(config.index_head_dim),
        norm_eps=float(config.rms_norm_eps),
    ).to(torch.bfloat16)
    index_compressed_kv = _rotate_compressed_prefill_kv(
        index_compressed_kv,
        config=config,
        layer=layer,
        seq_len=seq_len,
        start_pos=0,
        compress_ratio=compress_ratio,
    )
    return compressed_kv.contiguous(), index_compressed_kv.contiguous()


def _cache_state_mapping(cache_state: _LayerCacheState) -> dict[str, torch.Tensor]:
    return {
        "sliding_window_cache": cache_state.sliding_window_cache,
        "compressed_kv": cache_state.compressed_kv,
        "index_compressed_kv": cache_state.index_compressed_kv,
    }


def _prefill_layer_summary(
    asset: _LayerAssets,
    *,
    cache_state: _LayerCacheState,
    config: DeepSeekV4FlashConfig,
    materialized_prefill_output: bool,
) -> dict[str, Any]:
    return {
        "layer": int(asset.layer),
        "compress_ratio": int(config.compress_ratios[asset.layer]),
        "materialized_prefill_output": bool(materialized_prefill_output),
        "cache_after_prefill": _cache_summary(cache_state),
        "selected_experts": asset.prefill_router_preview,
        "fanout_scope": {
            "prefill_activated_expert_ids": [int(expert_id) for expert_id in asset.prefill_activated_experts],
            "prefill_activated_expert_count": len(asset.prefill_activated_experts),
            "prefill_loaded_expert_count": 0
            if asset.prefill_routed_weights_by_expert is None
            else len(asset.prefill_routed_weights_by_expert),
            "prefill_routes_executed": 0
            if asset.prefill_reference is None
            else int(asset.prefill_reference["ffn"]["router_indices"].numel()),
            "prefill_full_fanout_materialized": asset.prefill_reference is not None,
            "prefill_topk": int(config.num_experts_per_tok),
        },
        "output_shapes": {
            "prefill_post_ffn_residual": None
            if asset.prefill_reference is None
            else list(asset.prefill_reference["post_ffn_residual"].shape),
        },
    }


def _decode_layer_summary(
    *,
    layer: int,
    config: DeepSeekV4FlashConfig,
    current_position: int,
    cache_before: _LayerCacheState,
    cache_after: _LayerCacheState,
    reference: Mapping[str, Any],
    router_preview: Mapping[str, Any],
    activated_experts: Sequence[int],
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    attention = reference["attention"]
    sliding_len = int(attention["sliding_window_cache"].shape[1])
    compressed_topk_valid_count = int((attention["compress_topk_idxs"] >= sliding_len).sum().item())
    compressed_attention_delta_max = float(
        (attention["attention_output_rotary"].float() - attention["local_only_attention_output_rotary"].float())
        .abs()
        .max()
        .item()
    )
    activated_set = {int(expert_id) for expert_id in activated_experts}
    return {
        "layer": int(layer),
        "compress_ratio": int(config.compress_ratios[layer]),
        "current_position": int(current_position),
        "next_position": int(current_position + 1),
        "cache_before": _cache_summary(cache_before),
        "cache_after": _cache_summary(cache_after),
        "decode_cache": {
            "sliding_window_cache_before_decode": int(cache_before.sliding_window_cache.shape[1]),
            "sliding_window_cache_after_decode": int(attention["sliding_window_cache"].shape[1]),
            "current_token_cache_tokens": 1,
            "compressed_cache_length_used": int(attention["compressed_kv"].shape[1]),
            "compressed_cache_length_after_decode": int(cache_after.compressed_kv.shape[1]),
            "indexer_cache_length_after_decode": int(cache_after.index_compressed_kv.shape[1]),
            "attention_cache_length": int(attention["attention_cache"].shape[1]),
            "window_topk_valid_count": int((attention["window_topk_idxs"] >= 0).sum().item()),
            "compress_topk_valid_count": compressed_topk_valid_count,
            "runtime_topk_width": int(attention["runtime_topk_idxs"].shape[-1]),
            "indexer_topk_width": int(attention["indexer_topk_idxs"].shape[-1]),
            "compressed_tokens_contributed": compressed_topk_valid_count > 0 and compressed_attention_delta_max > 0.0,
            "compressed_attention_delta_max": compressed_attention_delta_max,
        },
        "selected_experts": {
            "decode": router_preview,
        },
        "fanout_scope": {
            "decode_activated_expert_ids": [int(expert_id) for expert_id in activated_experts],
            "decode_activated_expert_count": len(activated_experts),
            "decode_loaded_expert_count": len(routed_weights_by_expert),
            "decode_loaded_extra_candidate_count": len(set(routed_weights_by_expert) - activated_set),
            "decode_topk": int(config.num_experts_per_tok),
            "decode_routes_executed": int(reference["ffn"]["router_indices"].numel()),
            "sequential_ttnn_expert_execution": True,
        },
        "output_shapes": {
            "decode_input_hidden_states": list(reference["decode_input_hidden_states"].shape),
            "decode_attention_cache": list(attention["attention_cache"].shape),
            "decode_post_ffn_residual": list(reference["post_ffn_residual"].shape),
        },
    }


def _ttnn_layer_step_summary(layer_output: Mapping[str, Any]) -> dict[str, Any]:
    decode_outputs = layer_output["decode_outputs"]
    return {
        "layer": int(layer_output["layer"]),
        "decode_post_ffn_residual": _tensor_summary(decode_outputs["post_ffn_residual"]),
        "decode_attention_cache": _tensor_summary(decode_outputs["attention"]["attention_cache"]),
        "decode_runtime_topk_idxs": {
            "shape": list(decode_outputs["attention"]["runtime_topk_idxs"].shape),
            "valid_count": int((decode_outputs["attention"]["runtime_topk_idxs"] >= 0).sum().item()),
        },
        "decode_activated_experts": [int(expert_id) for expert_id in decode_outputs["per_expert_routes"].keys()],
        "decode_per_expert_routes": {
            str(expert_id): _selection_summary(route)
            for expert_id, route in decode_outputs["per_expert_routes"].items()
        },
        "decode_experts_executed": int(len(decode_outputs["per_expert_routes"])),
        "cache_after": _cache_summary(layer_output["cache_after"]),
    }


def _cache_summary(cache_state: _LayerCacheState) -> dict[str, Any]:
    return {
        "layer": int(cache_state.layer),
        "current_position": int(cache_state.current_position),
        "attention_input_tokens": int(cache_state.attention_input_history.shape[1]),
        "sliding_window_cache_length": int(cache_state.sliding_window_cache.shape[1]),
        "compressed_cache_length": int(cache_state.compressed_kv.shape[1]),
        "indexer_cache_length": int(cache_state.index_compressed_kv.shape[1]),
        "sliding_window_shape": list(cache_state.sliding_window_cache.shape),
        "compressed_shape": list(cache_state.compressed_kv.shape),
        "indexer_shape": list(cache_state.index_compressed_kv.shape),
    }


def _payload_bytes(
    metadata: Sequence[TensorMetadata],
    *,
    layer_assets: Sequence[_LayerAssets],
    logits_weights,
) -> dict[str, Any]:
    groups = _loaded_tensor_groups(metadata, layer_assets, logits_weights=logits_weights)
    return {
        "layers": {
            name: {
                sub_name: sub_group["payload_bytes"]
                for sub_name, sub_group in group.items()
                if isinstance(sub_group, dict) and "payload_bytes" in sub_group
            }
            for name, group in groups.items()
            if name.startswith("layer_")
        },
        "final_norm_lm_head": groups["final_norm_lm_head"]["payload_bytes"],
        "total": sum(item.nbytes for item in metadata),
    }


def _loaded_tensor_groups(
    metadata: Sequence[TensorMetadata],
    layer_assets: Sequence[_LayerAssets],
    *,
    logits_weights,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for asset in layer_assets:
        layer_groups = _metadata_groups(
            metadata,
            attention_keys=asset.attention_keys,
            selector_keys=asset.selector_keys,
            ffn_keys=asset.ffn_keys,
        )
        groups[f"layer_{asset.layer}"] = {
            name: {
                "count": len(items),
                "payload_bytes": sum(item.nbytes for item in items),
                "canonical_keys": [item.canonical_key for item in items],
            }
            for name, items in layer_groups.items()
        }
    groups["final_norm_lm_head"] = {
        "count": len(logits_weights.metadata),
        "payload_bytes": {
            "final_norm": sum(item.nbytes for item in logits_weights.metadata if item.canonical_key == "norm.weight"),
            "lm_head": sum(
                item.nbytes for item in logits_weights.metadata if item.canonical_key in ("head.weight", "embed.weight")
            ),
            "total": sum(item.nbytes for item in logits_weights.metadata),
        },
        "canonical_keys": [item.canonical_key for item in logits_weights.metadata],
    }
    return groups


def _host_boundaries(vocab_mode: VocabMode) -> list[dict[str, str]]:
    boundaries = [
        {
            "name": "embedding_weight_load",
            "location": "checkpoint load",
            "description": "real token embedding rows are loaded on host before the decode loop",
        },
        {
            "name": "embedding_lookup_host",
            "location": "before first decoder layer and between decode steps",
            "description": "supplied decode input_ids are embedded on host",
        },
        {
            "name": "projection_fp8_decode_to_bf16",
            "location": "checkpoint load",
            "description": "real FP8 attention and K/V weights are decoded on host to BF16 before TTNN modules",
        },
        {
            "name": "prefill_cache_handoff_readback",
            "location": "after prefill stack setup",
            "description": "layer cache state is host-visible before the multi-token decode loop",
        },
        {
            "name": "carried_decode_cache_host_state",
            "location": "between decode tokens",
            "description": "sliding-window KV, compressed KV, and indexer compressed KV are advanced on host",
        },
        {
            "name": "decode_indexer_host_topk",
            "location": "decode sparse attention",
            "description": "learned compressed-cache indexer top-k is computed at the host-visible cache boundary",
        },
        {
            "name": "sparse_attention_host_fallback",
            "location": "TtSparsePrefillAttention",
            "description": "indexed gather, attention-sink softmax, and weighted reduction currently run on host",
        },
        {
            "name": "router_topk",
            "location": "TtRouter",
            "description": "router scores leave device for host DeepSeek top-k/hash selection",
        },
        {
            "name": "decode_activated_expert_gather_scatter",
            "location": "decode FFN path",
            "description": "decode-token activated experts are gathered, executed sequentially, and scatter-added on host",
        },
        {
            "name": "prefill_activated_expert_gather_scatter",
            "location": "layer 2 prefill FFN path",
            "description": "layer-2 prefill activated experts are gathered, executed sequentially, and scatter-added on host",
        },
        {
            "name": "final_logits_readback",
            "location": "after LM head",
            "description": "full or sliced logits are copied back to host for dense accuracy and top-k comparison",
        },
    ]
    if vocab_mode == "slice":
        boundaries.append(
            {
                "name": "lm_head_vocab_slice",
                "location": "checkpoint load",
                "description": "a deterministic row slice of the LM head is loaded from safetensors on host",
            }
        )
    return boundaries


def _reference_ops(layers: Sequence[int], vocab_mode: VocabMode, decode_steps: int) -> list[str]:
    ops = ["torch.embedding(real_embed_weight)"]
    for layer_index, layer in enumerate(layers):
        prefix = f"layer_{int(layer)}"
        if layer_index != len(layers) - 1:
            ops.extend(
                [
                    f"{prefix}: torch.real_prefill_attention_runtime_reference",
                    f"{prefix}: torch.prefill_ffn_full_routed_fanout_reference",
                ]
            )
        else:
            ops.append(f"{prefix}: torch.prefill_attention_cache_reference")
    for _ in range(decode_steps):
        for layer in layers:
            prefix = f"layer_{int(layer)}"
            ops.extend(
                [
                    f"{prefix}: torch.real_decode_q_kv_projection_reference",
                    f"{prefix}: torch.sparse_attention_decode_reference(carried_cache)",
                    f"{prefix}: torch.decode_ffn_full_routed_fanout_reference",
                    f"{prefix}: torch.advance_decode_cache",
                ]
            )
        ops.extend(
            ["torch.rms_norm_reference(final_norm)", f"torch.linear(lm_head_{vocab_mode})", "torch.topk(logits)"]
        )
    return ops


def _ttnn_ops_summary(layers: Sequence[int], vocab_mode: VocabMode, decode_steps: int) -> list[str]:
    ops: list[str] = []
    for layer_index, layer in enumerate(layers):
        prefix = f"layer_{int(layer)}"
        if layer_index != len(layers) - 1:
            ops.extend(
                [
                    f"{prefix}: ttnn.rms_norm(prefill_attn_norm)",
                    f"{prefix}: TtAttentionProjection.project_q_rank(prefill)",
                    f"{prefix}: ttnn.linear(wkv, prefill)",
                    f"{prefix}: TtPrefillCompressor.build_compressed_kv_cache",
                    f"{prefix}: host_indexer_topk(prefill)",
                    f"{prefix}: TtSparsePrefillAttention(prefill)",
                    f"{prefix}: sequential_TtRoutedExpertMLP_per_activated_expert(prefill)",
                    f"{prefix}: TtSharedExpertMLP(prefill)",
                ]
            )
        else:
            ops.extend(
                [
                    f"{prefix}: ttnn.rms_norm(prefill_cache_only_attn_norm)",
                    f"{prefix}: TtAttentionProjection.project_q_rank(prefill_cache_only)",
                    f"{prefix}: ttnn.linear(wkv, prefill_cache_only)",
                ]
            )
    for _ in range(decode_steps):
        for layer in layers:
            prefix = f"layer_{int(layer)}"
            ops.extend(
                [
                    f"{prefix}: ttnn.rms_norm(decode_attn_norm)",
                    f"{prefix}: TtAttentionProjection.project_q_rank(decode)",
                    f"{prefix}: ttnn.linear(wkv, decode)",
                    f"{prefix}: host_rope_cache_prep(decode_position)",
                    f"{prefix}: host_decode_indexer_topk(carried_cache)",
                    f"{prefix}: TtSparsePrefillAttention(decode)",
                    f"{prefix}: ttnn.linear(wo_b)",
                    f"{prefix}: TtRouter(ttnn.linear+host_topk)",
                    f"{prefix}: sequential_TtRoutedExpertMLP_per_activated_expert(decode)",
                    f"{prefix}: TtSharedExpertMLP(decode)",
                    f"{prefix}: host_advance_decode_cache",
                ]
            )
        ops.extend(["ttnn.rms_norm(final_norm)", f"ttnn.linear(lm_head_{vocab_mode})"])
    return ops


def _not_materialized_prefill_summary(*, layer: int) -> dict[str, Any]:
    return {
        "source": "not_materialized",
        "layer": int(layer),
        "reason": "final stack layer only needs a prefill cache for the decode path",
    }


def _normalize_multi_input_ids(
    input_ids: Sequence[int] | torch.Tensor | None,
    *,
    prefill_seq_len: int,
    decode_steps: int,
    vocab_size: int,
    input_id_start: int,
) -> torch.Tensor:
    total_tokens = int(prefill_seq_len) + int(decode_steps)
    if input_ids is None:
        if input_id_start < 0:
            raise ValueError(f"input_id_start must be non-negative, got {input_id_start}")
        if input_id_start + total_tokens > int(vocab_size):
            raise ValueError(
                f"deterministic input ID range [{input_id_start}, {input_id_start + total_tokens}) "
                f"exceeds vocab size {vocab_size}"
            )
        values = torch.arange(input_id_start, input_id_start + total_tokens, dtype=torch.int64).reshape(1, -1)
    elif isinstance(input_ids, torch.Tensor):
        values = input_ids.to(torch.int64)
    else:
        values = torch.tensor([int(value) for value in input_ids], dtype=torch.int64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or tuple(values.shape) != (1, total_tokens):
        raise ValueError(f"input_ids must have shape [1, {total_tokens}], got {tuple(values.shape)}")
    if torch.any(values < 0) or torch.any(values >= int(vocab_size)):
        raise ValueError(f"input_ids values must be in [0, {vocab_size})")
    return values.contiguous()


def _validate_multi_token_args(
    *,
    layers: Sequence[int],
    prefill_seq_len: int,
    decode_steps: int,
    decode_feed_mode: DecodeFeedMode,
    max_tensors: int,
    max_bytes: int,
    pcc_thresholds: Mapping[str, float],
) -> None:
    if len(layers) < 2:
        raise ValueError("multi-token decode smoke requires at least two layers")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"layers must be non-negative, got {list(layers)}")
    if list(layers) != list(range(layers[0], layers[0] + len(layers))):
        raise ValueError(f"layers must be consecutive, got {list(layers)}")
    if prefill_seq_len <= 0:
        raise ValueError(f"prefill_seq_len must be positive, got {prefill_seq_len}")
    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be positive, got {decode_steps}")
    if decode_feed_mode != "supplied":
        raise ValueError(f"decode_feed_mode must be 'supplied', got {decode_feed_mode!r}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_multi_token_runtime_config(
    config: DeepSeekV4FlashConfig,
    *,
    layers: Sequence[int],
    prefill_seq_len: int,
    decode_steps: int,
) -> None:
    for layer in layers:
        if layer >= len(config.compress_ratios):
            raise ValueError(f"layer {layer} is outside compress_ratios length {len(config.compress_ratios)}")
    if int(config.num_key_value_heads) != 1:
        raise ValueError(f"Expected one K/V head, got {config.num_key_value_heads}")
    if prefill_seq_len + decode_steps > int(config.sliding_window):
        raise ValueError(
            f"multi-token decode smoke currently requires prefill_seq_len + decode_steps <= sliding_window "
            f"{config.sliding_window}, got {prefill_seq_len + decode_steps}"
        )


def _elapsed_seconds(start: float) -> float:
    return round(max(0.0, time.perf_counter() - start), 6)


def _sum_seconds(values: Iterable[float]) -> float:
    return round(sum(float(value) for value in values), 6)


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()
