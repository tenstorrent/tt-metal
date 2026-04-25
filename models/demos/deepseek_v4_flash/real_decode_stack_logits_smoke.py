# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import deterministic_attention_activation
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, TensorMetadata
from models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke import (
    DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES,
    DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS,
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
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights
from models.demos.deepseek_v4_flash.real_ffn_smoke import (
    _accuracy_summary,
    _metadata_summary,
    _selection_summary,
    _tensor_summary,
    build_torch_ffn_reference,
    layer_ffn_keys,
    validate_real_ffn_slice,
)
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import decode_real_kv_projection_weights
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES,
    PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE,
    _query_projection_weights_only,
    build_torch_prefill_attention_runtime_reference,
    decode_real_prefill_attention_projection_weights,
    decode_real_prefill_runtime_sparse_weights_if_needed,
    layer_prefill_attention_runtime_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    DEFAULT_SEQUENCE_LENGTH,
    build_torch_prefill_cache_prep_reference,
)
from models.demos.deepseek_v4_flash.real_prefill_decoder_layer_smoke import (
    _load_missing_tensors,
    _metadata_groups,
    _residual_add,
    _resolve_selected_expert,
    _run_ttnn_prefill_decoder_layer_slice,
    _unique_keys,
    build_torch_prefill_decoder_layer_reference,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_real_shared_expert_weights

REAL_DECODE_STACK_LOGITS_SMOKE_SCHEMA_VERSION = 1
DEFAULT_DECODE_STACK_LAYERS = (2, 3)
DEFAULT_DECODE_STACK_MAX_TENSORS = 2 * DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS + 12
DEFAULT_DECODE_STACK_MAX_BYTES = (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES + 2 * DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES + 2 * 1024 * 1024 * 1024
)


@dataclass(frozen=True)
class _LayerRuntime:
    layer: int
    q_weights: Any
    kv_weights: Any
    compressor_weights: Any
    indexer_weights: Any
    prefill_expert: int | None
    prefill_routed_weights: Mapping[str, torch.Tensor] | None
    prefill_shared_weights: Mapping[str, torch.Tensor] | None
    decode_expert: int
    decode_routed_weights: Mapping[str, torch.Tensor]
    decode_shared_weights: Mapping[str, torch.Tensor]
    prefill_cache_reference: Mapping[str, torch.Tensor]
    decode_cache_reference: Mapping[str, torch.Tensor]
    decode_attention_reference: Mapping[str, torch.Tensor]
    prefill_reference: Mapping[str, Any] | None
    decode_reference: Mapping[str, Any]
    attention_keys: Sequence[str]
    selector_keys: Sequence[str]
    ffn_keys: Sequence[str]


def run_real_decode_stack_logits_smoke(
    snapshot_dir: str | Path,
    *,
    layers: Sequence[int] = DEFAULT_DECODE_STACK_LAYERS,
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    vocab_mode: VocabMode = "full",
    vocab_start: int = 0,
    vocab_size: int | None = None,
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K,
    max_tensors: int = DEFAULT_DECODE_STACK_MAX_TENSORS,
    max_bytes: int = DEFAULT_DECODE_STACK_MAX_BYTES,
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
    """Run a tiny real consecutive-layer decode stack through final norm and LM head."""

    layers = tuple(int(layer) for layer in layers)
    _validate_stack_args(
        layers=layers,
        prefill_seq_len=prefill_seq_len,
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
    _validate_stack_runtime_config(config, layers=layers, prefill_seq_len=prefill_seq_len)
    current_position = int(prefill_seq_len)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)

    tensors: dict[str, torch.Tensor] = {}
    metadata: list[TensorMetadata] = []
    prefill_hidden = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=prefill_seq_len)
    decode_hidden = deterministic_attention_activation(
        hidden_size=config.hidden_size,
        seq_len=prefill_seq_len + 1,
    )[:, :, -1:, :].contiguous()

    layer_runtimes: list[_LayerRuntime] = []
    layer_summaries: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        is_last_layer = layer_index == len(layers) - 1
        layer_runtime, layer_summary, prefill_hidden, decode_hidden, metadata = _build_reference_layer_step(
            index,
            tensors,
            metadata,
            config=config,
            layer=layer,
            prefill_hidden=prefill_hidden,
            decode_hidden=decode_hidden,
            current_position=current_position,
            materialize_prefill_output=not is_last_layer,
            max_tensors=max_tensors,
            max_bytes=max_bytes,
        )
        layer_runtimes.append(layer_runtime)
        layer_summaries.append(layer_summary)

    logits_weights = load_decode_logits_weights(
        index,
        config=config,
        vocab_mode=vocab_mode,
        vocab_start=vocab_start,
        vocab_size=vocab_size,
        already_loaded_metadata=metadata,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    all_metadata = [*metadata, *logits_weights.metadata]
    logits_reference = build_torch_decode_logits_reference(
        decode_hidden,
        logits_weights.norm_weight,
        logits_weights.head_weight,
        config=config,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layers=layers,
        prefill_seq_len=prefill_seq_len,
        current_position=current_position,
        config=config,
        metadata=all_metadata,
        layer_runtimes=layer_runtimes,
        layer_summaries=layer_summaries,
        logits_weights=logits_weights,
        logits_reference=logits_reference,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        top_k=top_k,
    )

    if cpu_only:
        result.pop("_reference_tensors")
        result["mode"] = "cpu-reference"
        result["accuracy"] = {
            "cpu_reference": {
                "passed": True,
                "reason": "cpu-only requested; TTNN comparison was not run",
            }
        }
        result["passed"] = True
        return result

    if prefill_seq_len % PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(
            f"TTNN stack smoke prefill_seq_len must be a multiple of "
            f"{PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE}, got {prefill_seq_len}"
        )

    ttnn_outputs = _run_ttnn_stack(
        tensors,
        layer_runtimes,
        logits_weights,
        config=config,
        initial_prefill_hidden=result["_reference_tensors"]["initial_prefill_hidden"],
        initial_decode_hidden=result["_reference_tensors"]["initial_decode_hidden"],
        current_position=current_position,
        device_id=device_id,
    )
    result.pop("_reference_tensors")
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = _ttnn_ops_summary(layer_runtimes, logits_weights.vocab_mode)
    result["ttnn"] = {
        "layers": ttnn_outputs["layer_summaries"],
        "stack_hidden": _tensor_summary(ttnn_outputs["stack_hidden"]),
        "final_norm": _tensor_summary(ttnn_outputs["final_norm"]),
        "logits": _tensor_summary(ttnn_outputs["logits"]),
        "top_k": _topk_summary(
            ttnn_outputs["logits"],
            top_k=top_k,
            vocab_start=logits_weights.vocab_start,
        ),
    }
    accuracy = {
        f"layer_{int(layers[0])}_prefill_post_ffn_residual": _accuracy_summary(
            layer_runtimes[0].prefill_reference["post_ffn_residual"],
            ttnn_outputs["prefill_hidden_by_layer"][int(layers[0])],
            pcc_threshold=prefill_hidden_pcc,
            rtol=rtol,
            atol=residual_atol,
        ),
        "stack_hidden": _accuracy_summary(
            decode_hidden,
            ttnn_outputs["stack_hidden"],
            pcc_threshold=layer_hidden_pcc,
            rtol=rtol,
            atol=residual_atol,
        ),
        "final_norm": _accuracy_summary(
            logits_reference["final_norm"],
            ttnn_outputs["final_norm"],
            pcc_threshold=final_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "logits": _dense_pcc_summary(
            logits_reference["logits"],
            ttnn_outputs["logits"],
            pcc_threshold=logits_pcc,
            rtol=logits_rtol,
            atol=logits_atol,
        ),
        "top_k": _topk_accuracy_summary(
            logits_reference["logits"],
            ttnn_outputs["logits"],
            top_k=top_k,
            vocab_start=logits_weights.vocab_start,
            top_logit_atol=top_logit_atol,
        ),
    }
    for layer in layers:
        accuracy[f"layer_{layer}_decode_post_ffn_residual"] = _accuracy_summary(
            layer_runtimes[layers.index(layer)].decode_reference["post_ffn_residual"],
            ttnn_outputs["decode_hidden_by_layer"][int(layer)],
            pcc_threshold=layer_hidden_pcc,
            rtol=rtol,
            atol=residual_atol,
        )
    result["accuracy"] = accuracy
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real DeepSeek V4 Flash two-layer one-token decode stack through final norm and LM head."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_DECODE_STACK_LAYERS))
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--vocab-mode", choices=("full", "slice"), default="full")
    parser.add_argument("--vocab-start", type=int, default=0)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_DECODE_STACK_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_DECODE_STACK_MAX_BYTES)
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

    result = run_real_decode_stack_logits_smoke(
        args.snapshot_dir,
        layers=args.layers,
        prefill_seq_len=args.prefill_seq_len,
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
    result.pop("_reference_tensors", None)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _build_reference_layer_step(
    index: RealCheckpointTensorIndex,
    tensors: dict[str, torch.Tensor],
    metadata: list[TensorMetadata],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    prefill_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    current_position: int,
    materialize_prefill_output: bool,
    max_tensors: int,
    max_bytes: int,
) -> tuple[_LayerRuntime, dict[str, Any], torch.Tensor, torch.Tensor, list[TensorMetadata]]:
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
        seq_len=int(prefill_hidden.shape[2]),
    )

    prefill_reference = None
    prefill_expert = None
    prefill_router_preview = _not_materialized_prefill_summary(layer=layer)
    prefill_routed_weights = None
    prefill_shared_weights = None
    prefill_ffn_keys: list[str] = []
    if materialize_prefill_output:
        prefill_cache_reference = build_torch_prefill_attention_runtime_reference(
            tensors,
            q_weights,
            kv_weights,
            compressor_weights,
            indexer_weights,
            config=config,
            layer=layer,
            activation=prefill_hidden,
            start_pos=0,
        )
        prefill_post_attention_residual = _residual_add(
            prefill_hidden, prefill_cache_reference["attention_output_projected"]
        )
        prefill_expert, prefill_router_preview = _resolve_selected_expert(
            tensors,
            config=config,
            layer=layer,
            requested_expert=_default_requested_expert(tensors, layer=layer),
            post_attention_residual=prefill_post_attention_residual,
        )
        prefill_ffn_keys = layer_ffn_keys(index, layer=layer, expert=prefill_expert)
        tensors, metadata = _load_missing_tensors(
            index,
            tensors,
            metadata,
            prefill_ffn_keys,
            max_tensors=max_tensors,
            max_bytes=max_bytes,
        )
        validate_real_ffn_slice(tensors, config=config, layer=layer, expert=prefill_expert)
        prefill_routed_weights = decode_real_expert_weights(tensors, config=config, layer=layer, expert=prefill_expert)
        prefill_shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=layer)
        prefill_reference = build_torch_prefill_decoder_layer_reference(
            tensors,
            q_weights,
            kv_weights,
            compressor_weights,
            indexer_weights,
            prefill_routed_weights,
            prefill_shared_weights,
            config=config,
            layer=layer,
            expert=prefill_expert,
            activation=prefill_hidden,
            start_pos=0,
            attention_reference=prefill_cache_reference,
        )
        next_prefill_hidden = prefill_reference["post_ffn_residual"]
    else:
        prefill_cache_reference = build_torch_prefill_cache_prep_reference(
            tensors,
            _query_projection_weights_only(q_weights),
            kv_weights,
            config=config,
            layer=layer,
            activation=prefill_hidden,
            start_pos=0,
        )
        next_prefill_hidden = prefill_hidden

    decode_cache_reference = build_torch_decode_cache_prep_reference(
        tensors,
        _query_projection_weights_only(q_weights),
        kv_weights,
        config=config,
        layer=layer,
        activation=decode_hidden,
        current_position=current_position,
    )
    decode_attention_reference = build_torch_decode_attention_runtime_reference(
        prefill_cache_reference,
        decode_cache_reference,
        q_weights,
        tensors[f"layers.{layer}.attn.attn_sink"],
        config=config,
        layer=layer,
        current_position=current_position,
        indexer_weights=indexer_weights,
    )
    decode_post_attention_residual = _residual_add(
        decode_hidden, decode_attention_reference["attention_output_projected"]
    )
    decode_expert, decode_router_preview = _resolve_selected_expert(
        tensors,
        config=config,
        layer=layer,
        requested_expert=_default_requested_expert(tensors, layer=layer),
        post_attention_residual=decode_post_attention_residual,
    )
    decode_ffn_keys = layer_ffn_keys(index, layer=layer, expert=decode_expert)
    tensors, metadata = _load_missing_tensors(
        index,
        tensors,
        metadata,
        decode_ffn_keys,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    validate_real_ffn_slice(tensors, config=config, layer=layer, expert=decode_expert)
    decode_routed_weights = decode_real_expert_weights(tensors, config=config, layer=layer, expert=decode_expert)
    decode_shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=layer)
    decode_ffn_reference = build_torch_ffn_reference(
        tensors,
        decode_routed_weights,
        decode_shared_weights,
        config=config,
        layer=layer,
        expert=decode_expert,
        activation=decode_post_attention_residual,
    )
    decode_reference = {
        "decode_input_hidden_states": decode_hidden,
        "decode_cache": decode_cache_reference,
        "attention": decode_attention_reference,
        "post_attention_residual": decode_post_attention_residual,
        "ffn": decode_ffn_reference,
        "post_ffn_residual": decode_ffn_reference["residual_output"],
    }
    ffn_keys = _unique_keys([*prefill_ffn_keys, *decode_ffn_keys])
    runtime = _LayerRuntime(
        layer=layer,
        q_weights=q_weights,
        kv_weights=kv_weights,
        compressor_weights=compressor_weights,
        indexer_weights=indexer_weights,
        prefill_expert=prefill_expert,
        prefill_routed_weights=prefill_routed_weights,
        prefill_shared_weights=prefill_shared_weights,
        decode_expert=decode_expert,
        decode_routed_weights=decode_routed_weights,
        decode_shared_weights=decode_shared_weights,
        prefill_cache_reference=prefill_cache_reference,
        decode_cache_reference=decode_cache_reference,
        decode_attention_reference=decode_attention_reference,
        prefill_reference=prefill_reference,
        decode_reference=decode_reference,
        attention_keys=attention_keys,
        selector_keys=selector_keys,
        ffn_keys=ffn_keys,
    )
    summary = _layer_summary(
        runtime,
        config=config,
        prefill_router_preview=prefill_router_preview,
        decode_router_preview=decode_router_preview,
        current_position=current_position,
        materialized_prefill_output=materialize_prefill_output,
    )
    return runtime, summary, next_prefill_hidden, decode_reference["post_ffn_residual"], metadata


def _run_ttnn_stack(
    tensors: Mapping[str, torch.Tensor],
    layer_runtimes: Sequence[_LayerRuntime],
    logits_weights,
    *,
    config: DeepSeekV4FlashConfig,
    initial_prefill_hidden: torch.Tensor,
    initial_decode_hidden: torch.Tensor,
    current_position: int,
    device_id: int,
) -> dict[str, Any]:
    prefill_hidden = initial_prefill_hidden
    decode_hidden = initial_decode_hidden
    layer_summaries: list[dict[str, Any]] = []
    prefill_hidden_by_layer: dict[int, torch.Tensor] = {}
    decode_hidden_by_layer: dict[int, torch.Tensor] = {}
    for layer_index, runtime in enumerate(layer_runtimes):
        is_last_layer = layer_index == len(layer_runtimes) - 1
        if runtime.prefill_routed_weights is not None and runtime.prefill_shared_weights is not None:
            prefill_outputs = _run_ttnn_prefill_decoder_layer_slice(
                tensors,
                runtime.q_weights,
                runtime.kv_weights,
                runtime.compressor_weights,
                runtime.indexer_weights,
                runtime.prefill_routed_weights,
                runtime.prefill_shared_weights,
                config=config,
                layer=runtime.layer,
                expert=int(runtime.prefill_expert),
                activation=prefill_hidden,
                start_pos=0,
                device_id=device_id,
            )
            prefill_hidden = prefill_outputs["post_ffn_residual"]
            prefill_hidden_by_layer[int(runtime.layer)] = prefill_hidden
            prefill_cache_outputs = prefill_outputs
        else:
            prefill_outputs = None
            prefill_cache_outputs = None

        decode_outputs = _run_ttnn_decode_decoder_layer_slice(
            tensors,
            runtime.q_weights,
            runtime.kv_weights,
            runtime.decode_routed_weights,
            runtime.decode_shared_weights,
            config=config,
            layer=runtime.layer,
            expert=runtime.decode_expert,
            prefill_activation=prefill_hidden,
            decode_activation=decode_hidden,
            current_position=current_position,
            device_id=device_id,
            prefill_cache_outputs=prefill_cache_outputs,
            indexer_weights=runtime.indexer_weights,
        )
        decode_hidden = decode_outputs["post_ffn_residual"]
        decode_hidden_by_layer[int(runtime.layer)] = decode_hidden
        layer_summaries.append(_ttnn_layer_summary(runtime.layer, prefill_outputs, decode_outputs))
        if is_last_layer:
            break

    logits_outputs = _run_ttnn_decode_logits_from_hidden(
        decode_hidden,
        logits_weights.norm_weight,
        logits_weights.head_weight,
        config=config,
        device_id=device_id,
    )
    return {
        "layer_summaries": layer_summaries,
        "prefill_hidden_by_layer": prefill_hidden_by_layer,
        "decode_hidden_by_layer": decode_hidden_by_layer,
        "stack_hidden": decode_hidden,
        "final_norm": logits_outputs["final_norm"],
        "logits": logits_outputs["logits"],
    }


def _base_result(
    *,
    snapshot_dir: Path,
    layers: Sequence[int],
    prefill_seq_len: int,
    current_position: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    layer_runtimes: Sequence[_LayerRuntime],
    layer_summaries: Sequence[dict[str, Any]],
    logits_weights,
    logits_reference: Mapping[str, torch.Tensor],
    max_tensors: int,
    max_bytes: int,
    top_k: int,
) -> dict[str, Any]:
    payload_bytes = _payload_bytes(metadata, layer_runtimes=layer_runtimes, logits_weights=logits_weights)
    stack_hidden = layer_runtimes[-1].decode_reference["post_ffn_residual"]
    return {
        "schema_version": REAL_DECODE_STACK_LOGITS_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layers": [int(layer) for layer in layers],
        "prefill_sequence_length": int(prefill_seq_len),
        "decode_tokens": 1,
        "current_position": int(current_position),
        "next_position": int(current_position + 1),
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
        "stack_scope": {
            "path": (
                "deterministic hidden -> layer 2 prefill cache and one-token decode -> "
                "layer 3 prefill cache from layer 2 prefill output and one-token decode -> final norm -> LM head"
            ),
            "prefill_handoff": "layer 3 prefill cache is built from the TTNN/torch layer 2 prefill decoder output",
            "decode_handoff": "layer 2 one-token decode post-FFN residual feeds layer 3 one-token decode",
            "full_expert_fanout": "excluded; each materialized FFN uses the selected routed expert plus shared expert",
            "embeddings_vllm_evals": "excluded",
        },
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "loaded_tensor_groups": _loaded_tensor_groups(metadata, layer_runtimes, logits_weights=logits_weights),
        "payload_bytes": payload_bytes,
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": payload_bytes["total"],
        },
        "cache_handoff": {
            "layer_3_prefill_cache_source": "layer_2_prefill_post_ffn_residual",
            "host_visible_boundary": True,
            "temporary_boundary": (
                "hidden states and cache tensors are read back between layer helpers until model.py owns device-resident "
                "multi-layer cache handoff"
            ),
        },
        "layers_detail": list(layer_summaries),
        "layer2_compressed_tokens_contributed": bool(
            layer_summaries[0]["decode_cache"]["compressed_tokens_contributed"]
        ),
        "output_shapes": {
            "initial_prefill_hidden": [1, 1, int(prefill_seq_len), int(config.hidden_size)],
            "initial_decode_hidden": [1, 1, 1, int(config.hidden_size)],
            "stack_hidden": list(stack_hidden.shape),
            "final_norm": list(logits_reference["final_norm"].shape),
            "logits": list(logits_reference["logits"].shape),
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
        "reference_ops": _reference_ops(layer_runtimes, logits_weights.vocab_mode),
        "ttnn_ops": [],
        "reference": {
            "layers": [
                {
                    "layer": runtime.layer,
                    "prefill_post_ffn_residual": (
                        None
                        if runtime.prefill_reference is None
                        else _tensor_summary(runtime.prefill_reference["post_ffn_residual"])
                    ),
                    "decode_post_ffn_residual": _tensor_summary(runtime.decode_reference["post_ffn_residual"]),
                }
                for runtime in layer_runtimes
            ],
            "stack_hidden": _tensor_summary(stack_hidden),
            "final_norm": _tensor_summary(logits_reference["final_norm"]),
            "logits": _tensor_summary(logits_reference["logits"]),
            "top_k": _topk_summary(logits_reference["logits"], top_k=top_k, vocab_start=logits_weights.vocab_start),
        },
        "ttnn": {},
        "accuracy": {},
        "passed": False,
        "_reference_tensors": {
            "initial_prefill_hidden": layer_runtimes[0].prefill_reference["input_hidden_states"],
            "initial_decode_hidden": layer_runtimes[0].decode_reference["decode_input_hidden_states"],
        },
    }


def _layer_summary(
    runtime: _LayerRuntime,
    *,
    config: DeepSeekV4FlashConfig,
    prefill_router_preview: Mapping[str, Any],
    decode_router_preview: Mapping[str, Any],
    current_position: int,
    materialized_prefill_output: bool,
) -> dict[str, Any]:
    attention = runtime.decode_attention_reference
    prefill_cache = runtime.prefill_cache_reference
    sliding_len = int(attention["sliding_window_cache"].shape[1])
    compressed_topk_valid_count = int((attention["compress_topk_idxs"] >= sliding_len).sum().item())
    compressed_attention_delta_max = float(
        (attention["attention_output_rotary"].float() - attention["local_only_attention_output_rotary"].float())
        .abs()
        .max()
        .item()
    )
    return {
        "layer": int(runtime.layer),
        "compress_ratio": int(config.compress_ratios[runtime.layer]),
        "current_position": int(current_position),
        "next_position": int(current_position + 1),
        "materialized_prefill_output": bool(materialized_prefill_output),
        "prefill_cache": {
            "input_hidden_shape": list(
                runtime.prefill_reference["input_hidden_states"].shape
                if runtime.prefill_reference is not None
                else prefill_cache["attn_norm_output"].shape
            ),
            "sliding_window_cache_length": int(prefill_cache["sliding_window_cache"].shape[1]),
            "compressed_cache_length": int(prefill_cache.get("compressed_kv", torch.empty(1, 0, 1)).shape[1]),
            "indexer_cache_length": int(prefill_cache.get("index_compressed_kv", torch.empty(1, 0, 1)).shape[1]),
        },
        "decode_cache": {
            "sliding_window_cache_before_decode": int(prefill_cache["sliding_window_cache"].shape[1]),
            "sliding_window_cache_after_decode": int(attention["sliding_window_cache"].shape[1]),
            "current_token_cache_tokens": 1,
            "compressed_cache_length": int(attention["compressed_kv"].shape[1]),
            "attention_cache_length": int(attention["attention_cache"].shape[1]),
            "window_topk_valid_count": int((attention["window_topk_idxs"] >= 0).sum().item()),
            "compress_topk_valid_count": compressed_topk_valid_count,
            "runtime_topk_width": int(attention["runtime_topk_idxs"].shape[-1]),
            "indexer_topk_width": int(attention["indexer_topk_idxs"].shape[-1]),
            "compressed_tokens_contributed": compressed_topk_valid_count > 0 and compressed_attention_delta_max > 0.0,
            "compressed_attention_delta_max": compressed_attention_delta_max,
        },
        "selected_experts": {
            "prefill": prefill_router_preview,
            "decode": decode_router_preview,
        },
        "output_shapes": {
            "prefill_post_ffn_residual": (
                None
                if runtime.prefill_reference is None
                else list(runtime.prefill_reference["post_ffn_residual"].shape)
            ),
            "decode_input_hidden_states": list(runtime.decode_reference["decode_input_hidden_states"].shape),
            "decode_attention_cache": list(attention["attention_cache"].shape),
            "decode_post_ffn_residual": list(runtime.decode_reference["post_ffn_residual"].shape),
        },
    }


def _ttnn_layer_summary(
    layer: int,
    prefill_outputs: Mapping[str, Any] | None,
    decode_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "layer": int(layer),
        "prefill_post_ffn_residual": (
            None if prefill_outputs is None else _tensor_summary(prefill_outputs["post_ffn_residual"])
        ),
        "decode_post_ffn_residual": _tensor_summary(decode_outputs["post_ffn_residual"]),
        "decode_attention_cache": _tensor_summary(decode_outputs["attention"]["attention_cache"]),
        "decode_runtime_topk_idxs": {
            "shape": list(decode_outputs["attention"]["runtime_topk_idxs"].shape),
            "valid_count": int((decode_outputs["attention"]["runtime_topk_idxs"] >= 0).sum().item()),
        },
        "decode_selected_route": _selection_summary(decode_outputs["selected_route"]),
    }


def _payload_bytes(
    metadata: Sequence[TensorMetadata],
    *,
    layer_runtimes: Sequence[_LayerRuntime],
    logits_weights,
) -> dict[str, Any]:
    groups = _loaded_tensor_groups(metadata, layer_runtimes, logits_weights=logits_weights)
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
    layer_runtimes: Sequence[_LayerRuntime],
    *,
    logits_weights,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for runtime in layer_runtimes:
        layer_groups = _metadata_groups(
            metadata,
            attention_keys=runtime.attention_keys,
            selector_keys=runtime.selector_keys,
            ffn_keys=runtime.ffn_keys,
        )
        groups[f"layer_{runtime.layer}"] = {
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
            "name": "projection_fp8_decode_to_bf16",
            "location": "checkpoint load",
            "description": "real FP8 attention and K/V weights are decoded on host to BF16 before TTNN modules",
        },
        {
            "name": "layer2_prefill_output_readback",
            "location": "between layer 2 prefill and layer 3 cache prep",
            "description": "layer 2 prefill post-FFN residual is host-visible before becoming layer 3 prefill input",
        },
        {
            "name": "layer2_decode_output_readback",
            "location": "between layer 2 decode and layer 3 decode",
            "description": "layer 2 one-token post-FFN residual is host-visible before becoming layer 3 decode input",
        },
        {
            "name": "layer2_decode_indexer_host_topk",
            "location": "layer 2 decode attention",
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
            "description": "router scores leave device for host DeepSeek top-k selection",
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


def _reference_ops(layer_runtimes: Sequence[_LayerRuntime], vocab_mode: VocabMode) -> list[str]:
    ops: list[str] = []
    for runtime in layer_runtimes:
        prefix = f"layer_{runtime.layer}"
        if runtime.prefill_reference is not None:
            ops.extend(
                [
                    f"{prefix}: torch.real_prefill_attention_runtime_reference",
                    f"{prefix}: torch.prefill_ffn_reference",
                ]
            )
        else:
            ops.append(f"{prefix}: torch.prefill_cache_prep_reference")
        ops.extend(
            [
                f"{prefix}: torch.real_decode_q_kv_projection_reference",
                f"{prefix}: torch.sparse_attention_decode_reference",
                f"{prefix}: torch.decode_ffn_reference",
            ]
        )
    ops.extend(["torch.rms_norm_reference(final_norm)", f"torch.linear(lm_head_{vocab_mode})", "torch.topk(logits)"])
    return ops


def _ttnn_ops_summary(layer_runtimes: Sequence[_LayerRuntime], vocab_mode: VocabMode) -> list[str]:
    ops: list[str] = []
    for runtime in layer_runtimes:
        prefix = f"layer_{runtime.layer}"
        if runtime.prefill_reference is not None:
            ops.extend(
                [
                    f"{prefix}: ttnn.rms_norm(prefill_attn_norm)",
                    f"{prefix}: TtAttentionProjection.project_q_rank(prefill)",
                    f"{prefix}: ttnn.linear(wkv, prefill)",
                    f"{prefix}: TtPrefillCompressor.build_compressed_kv_cache",
                    f"{prefix}: host_indexer_topk(prefill)",
                    f"{prefix}: TtSparsePrefillAttention(prefill)",
                    f"{prefix}: TtRoutedExpertMLP(prefill)",
                    f"{prefix}: TtSharedExpertMLP(prefill)",
                ]
            )
        ops.extend(
            [
                f"{prefix}: ttnn.rms_norm(decode_attn_norm)",
                f"{prefix}: TtAttentionProjection.project_q_rank(decode)",
                f"{prefix}: ttnn.linear(wkv, decode)",
                f"{prefix}: host_rope_cache_prep(decode_position)",
                f"{prefix}: host_decode_indexer_topk",
                f"{prefix}: TtSparsePrefillAttention(decode)",
                f"{prefix}: ttnn.linear(wo_b)",
                f"{prefix}: TtRouter(ttnn.linear+host_topk)",
                f"{prefix}: TtRoutedExpertMLP(decode)",
                f"{prefix}: TtSharedExpertMLP(decode)",
            ]
        )
    ops.extend(["ttnn.rms_norm(final_norm)", f"ttnn.linear(lm_head_{vocab_mode})"])
    return ops


def _not_materialized_prefill_summary(*, layer: int) -> dict[str, Any]:
    return {
        "source": "not_materialized",
        "layer": int(layer),
        "reason": "final stack layer only needs a prefill cache for the one-token decode path",
    }


def _default_requested_expert(tensors: Mapping[str, torch.Tensor], *, layer: int) -> int | None:
    if tensors.get(f"layers.{layer}.ffn.gate.tid2eid") is None:
        return None
    return 0


def _validate_stack_args(
    *,
    layers: Sequence[int],
    prefill_seq_len: int,
    max_tensors: int,
    max_bytes: int,
    pcc_thresholds: Mapping[str, float],
) -> None:
    if len(layers) < 2:
        raise ValueError("decode stack smoke requires at least two layers")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"layers must be non-negative, got {list(layers)}")
    if list(layers) != list(range(layers[0], layers[0] + len(layers))):
        raise ValueError(f"layers must be consecutive, got {list(layers)}")
    if prefill_seq_len <= 0:
        raise ValueError(f"prefill_seq_len must be positive, got {prefill_seq_len}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_stack_runtime_config(
    config: DeepSeekV4FlashConfig,
    *,
    layers: Sequence[int],
    prefill_seq_len: int,
) -> None:
    for layer in layers:
        if layer >= len(config.compress_ratios):
            raise ValueError(f"layer {layer} is outside compress_ratios length {len(config.compress_ratios)}")
    if int(config.num_key_value_heads) != 1:
        raise ValueError(f"Expected one K/V head, got {config.num_key_value_heads}")
    if prefill_seq_len + 1 > int(config.sliding_window):
        raise ValueError(
            f"decode stack smoke requires prefill_seq_len + 1 <= sliding_window {config.sliding_window}, "
            f"got {prefill_seq_len + 1}"
        )


if __name__ == "__main__":
    main()
