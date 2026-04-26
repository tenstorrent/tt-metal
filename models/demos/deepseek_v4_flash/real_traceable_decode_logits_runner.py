# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Eval-shaped wrapper for the protected carried-cache traceable decode layer.

The protected region is still the one-layer traceable decode body. Embedding
lookup, checkpoint decode, final RMSNorm, sliced/full LM head, top-k, and JSON
reporting intentionally stay outside that region.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_decode_logits_smoke import (
    DEFAULT_DECODE_LOGITS_TOP_K,
    DEFAULT_DECODE_LOGITS_VOCAB_SLICE_SIZE,
    VocabMode,
    _dense_pcc_summary,
    _run_ttnn_decode_logits_from_hidden,
    _topk_accuracy_summary,
    _topk_summary,
    _validate_logits_args,
    build_torch_decode_logits_reference,
    load_decode_logits_weights,
)
from models.demos.deepseek_v4_flash.real_ffn_smoke import _accuracy_summary, _metadata_summary, _tensor_summary
from models.demos.deepseek_v4_flash.real_generation_eval_runner import DEFAULT_REAL_SNAPSHOT_DIR, _resolve_input
from models.demos.deepseek_v4_flash.real_input_stack_logits_smoke import (
    DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS,
    EmbeddingMode,
    InputEmbeddingPayload,
    load_input_embedding_payload,
)
from models.demos.deepseek_v4_flash.real_multi_token_decode_smoke import _normalize_multi_input_ids
from models.demos.deepseek_v4_flash.real_traceable_decode_smoke import (
    DEFAULT_TRACEABLE_DECODE_MAX_BYTES,
    DEFAULT_TRACEABLE_DECODE_MAX_TENSORS,
    DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE,
    TRACEABLE_DECODE_ATTENTION_MODES,
    TRACEABLE_DECODE_ATTENTION_READ_APIS,
    TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
    TRACEABLE_DECODE_CACHE_UPDATE_APIS,
    TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
    TRACEABLE_DECODE_COMPRESSOR_MODES,
    TRACEABLE_DECODE_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    TRACEABLE_DECODE_INDEXER_COMPRESSOR_MODES,
    TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    TRACEABLE_DECODE_ROPE_POSITION_APIS,
    TRACEABLE_DECODE_ROPE_POSITION_STATIC,
    TRACEABLE_DECODE_SPARSE_INDEXER_MODES,
    TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
    run_traceable_decode_subpath_smoke,
)

REAL_TRACEABLE_DECODE_LOGITS_RUNNER_SCHEMA_VERSION = 1
RUNNER_NAME = "deepseek_v4_flash_real_traceable_decode_logits_runner"
DEFAULT_TRACEABLE_DECODE_LOGITS_LAYER = 4
DEFAULT_TRACEABLE_DECODE_LOGITS_PREFILL_SEQ_LEN = 32
DEFAULT_TRACEABLE_DECODE_LOGITS_STEPS = 4
DEFAULT_TRACEABLE_DECODE_LOGITS_CACHE_LEN = 96
DEFAULT_TRACEABLE_DECODE_LOGITS_MAX_TENSORS = DEFAULT_TRACEABLE_DECODE_MAX_TENSORS + 32
DEFAULT_TRACEABLE_DECODE_LOGITS_MAX_BYTES = DEFAULT_TRACEABLE_DECODE_MAX_BYTES + 2 * 1024 * 1024 * 1024


def run_real_traceable_decode_logits_runner(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_TRACEABLE_DECODE_LOGITS_LAYER,
    prefill_seq_len: int = DEFAULT_TRACEABLE_DECODE_LOGITS_PREFILL_SEQ_LEN,
    decode_steps: int = DEFAULT_TRACEABLE_DECODE_LOGITS_STEPS,
    input_ids: Sequence[int] | None = None,
    input_ids_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    input_id_start: int = 0,
    prompt_label: str | None = None,
    embedding_mode: EmbeddingMode = "slice",
    max_embedding_rows: int = DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS,
    vocab_mode: VocabMode = "slice",
    full_vocab_smoke: bool = False,
    vocab_start: int = 0,
    vocab_size: int | None = DEFAULT_DECODE_LOGITS_VOCAB_SLICE_SIZE,
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K,
    max_tensors: int = DEFAULT_TRACEABLE_DECODE_LOGITS_MAX_TENSORS,
    max_bytes: int = DEFAULT_TRACEABLE_DECODE_LOGITS_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    trace_region_size: int = DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE,
    cache_len: int = DEFAULT_TRACEABLE_DECODE_LOGITS_CACHE_LEN,
    attention_mode: str = "traceable_fixed_cache_window_qk_softmax",
    attention_read_api: str = TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
    cache_update_api: str = TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
    rope_position_api: str = TRACEABLE_DECODE_ROPE_POSITION_STATIC,
    sparse_indexer_mode: str = TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
    compressor_mode: str = TRACEABLE_DECODE_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    indexer_compressor_mode: str = TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    routed_topk_prefix: int | None = None,
    trace_pcc: float = 0.99,
    final_norm_pcc: float = 0.999,
    logits_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    logits_rtol: float = 1e-1,
    logits_atol: float = 1.0,
    top_logit_atol: float = 1.0,
) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    total_tokens = int(prefill_seq_len) + int(decode_steps)
    _validate_runner_args(
        config=config,
        layer=layer,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
        cache_len=cache_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    effective_vocab_mode = "full" if full_vocab_smoke else vocab_mode
    effective_vocab_start = 0 if full_vocab_smoke else vocab_start
    effective_vocab_size = None if full_vocab_smoke else vocab_size
    _validate_logits_args(
        vocab_mode=effective_vocab_mode,
        vocab_start=effective_vocab_start,
        vocab_size=effective_vocab_size,
        top_k=top_k,
    )

    wall_start = time.perf_counter()
    resolved_input = _resolve_input(
        snapshot_dir=snapshot_dir,
        total_tokens=total_tokens,
        input_ids=input_ids,
        input_ids_path=input_ids_path,
        prompt_path=prompt_path,
        prompt_label=prompt_label,
    )
    token_ids = _normalize_multi_input_ids(
        resolved_input.input_ids,
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
    step_activations = _build_real_embedding_trace_activations(
        embedding.hidden_states,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
    )

    trace_result = run_traceable_decode_subpath_smoke(
        snapshot_dir,
        layer=layer,
        seq_len=prefill_seq_len,
        routed_topk_prefix=routed_topk_prefix,
        max_tensors=max_tensors - 1,
        max_bytes=max_bytes - int(embedding.metadata.nbytes),
        cpu_only=cpu_only,
        device_id=device_id,
        trace_region_size=trace_region_size,
        cache_len=cache_len,
        cache_update_index=prefill_seq_len,
        decode_steps=decode_steps,
        attention_mode=attention_mode,
        attention_read_api=attention_read_api,
        cache_update_api=cache_update_api,
        rope_position_api=rope_position_api,
        sparse_indexer_mode=sparse_indexer_mode,
        compressor_mode=compressor_mode,
        indexer_compressor_mode=indexer_compressor_mode,
        pcc=trace_pcc,
        rtol=rtol,
        atol=atol,
        activation=step_activations[0],
        step_activations=step_activations,
        step_replay_activations=step_activations,
        activation_source="real_input_ids_host_embedding_trace_envelope",
        include_private_tensors=True,
    )
    reference_steps = trace_result.pop("_replay_reference_tensors_by_step")
    ttnn_steps = trace_result.pop("_ttnn_tensors_by_step", None)
    trace_result.pop("_reference_tensors_by_step", None)

    remaining_tensors = int(max_tensors) - 1 - len(trace_result["loaded_tensors"])
    remaining_bytes = int(max_bytes) - int(embedding.metadata.nbytes) - int(trace_result["payload_bytes"]["total"])
    if remaining_tensors <= 0:
        raise ValueError("max_tensors does not leave room for final norm and LM-head tensors")
    if remaining_bytes <= 0:
        raise ValueError("max_bytes does not leave room for final norm and LM-head tensors")
    logits_weights = load_decode_logits_weights(
        index,
        config=config,
        vocab_mode=effective_vocab_mode,  # type: ignore[arg-type]
        vocab_start=effective_vocab_start,
        vocab_size=effective_vocab_size,
        already_loaded_metadata=(),
        max_tensors=remaining_tensors,
        max_bytes=remaining_bytes,
    )
    logits_result = _run_logits_steps(
        reference_steps,
        ttnn_steps=ttnn_steps,
        logits_weights=logits_weights,
        config=config,
        top_k=top_k,
        cpu_only=cpu_only,
        device_id=device_id,
        final_norm_pcc=final_norm_pcc,
        logits_pcc=logits_pcc,
        rtol=rtol,
        atol=atol,
        logits_rtol=logits_rtol,
        logits_atol=logits_atol,
        top_logit_atol=top_logit_atol,
    )
    result = _build_result(
        snapshot_dir=snapshot_dir,
        config=config,
        trace_result=trace_result,
        logits_result=logits_result,
        embedding=embedding,
        logits_weights=logits_weights,
        input_ids=token_ids,
        resolved_input=resolved_input,
        full_vocab_smoke=full_vocab_smoke,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        wall_seconds=_seconds_since(wall_start),
    )
    return result


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real input IDs through the protected carried-cache traceable decode layer and logits head."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(DEFAULT_REAL_SNAPSHOT_DIR))),
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_TRACEABLE_DECODE_LOGITS_LAYER)
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_TRACEABLE_DECODE_LOGITS_PREFILL_SEQ_LEN)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_TRACEABLE_DECODE_LOGITS_STEPS)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--input-ids-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--input-id-start", type=int, default=0)
    parser.add_argument("--prompt-label")
    parser.add_argument("--embedding-mode", choices=("slice", "full"), default="slice")
    parser.add_argument("--max-embedding-rows", type=int, default=DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS)
    parser.add_argument(
        "--vocab-mode",
        choices=("full", "slice"),
        default=os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_VOCAB_MODE", "slice"),
    )
    parser.add_argument("--full-vocab-smoke", action="store_true")
    parser.add_argument(
        "--vocab-start", type=int, default=_optional_int_env("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_VOCAB_START", 0)
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_optional_int_env(
            "DSV4_FLASH_TRACEABLE_DECODE_LOGITS_VOCAB_SIZE",
            DEFAULT_DECODE_LOGITS_VOCAB_SLICE_SIZE,
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_TRACEABLE_DECODE_LOGITS_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_TRACEABLE_DECODE_LOGITS_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE)
    parser.add_argument("--cache-len", type=int, default=DEFAULT_TRACEABLE_DECODE_LOGITS_CACHE_LEN)
    parser.add_argument(
        "--attention-mode", choices=TRACEABLE_DECODE_ATTENTION_MODES, default="traceable_fixed_cache_window_qk_softmax"
    )
    parser.add_argument(
        "--attention-read-api",
        choices=TRACEABLE_DECODE_ATTENTION_READ_APIS,
        default=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
    )
    parser.add_argument(
        "--cache-update-api",
        choices=TRACEABLE_DECODE_CACHE_UPDATE_APIS,
        default=TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
    )
    parser.add_argument(
        "--rope-position-api",
        choices=TRACEABLE_DECODE_ROPE_POSITION_APIS,
        default=TRACEABLE_DECODE_ROPE_POSITION_STATIC,
    )
    parser.add_argument(
        "--sparse-indexer-mode",
        choices=TRACEABLE_DECODE_SPARSE_INDEXER_MODES,
        default=TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
    )
    parser.add_argument(
        "--compressor-mode",
        choices=TRACEABLE_DECODE_COMPRESSOR_MODES,
        default=TRACEABLE_DECODE_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    )
    parser.add_argument(
        "--indexer-compressor-mode",
        choices=TRACEABLE_DECODE_INDEXER_COMPRESSOR_MODES,
        default=TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    )
    parser.add_argument("--routed-topk-prefix", type=int)
    parser.add_argument("--trace-pcc", type=float, default=0.99)
    parser.add_argument("--final-norm-pcc", type=float, default=0.999)
    parser.add_argument("--logits-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--logits-rtol", type=float, default=1e-1)
    parser.add_argument("--logits-atol", type=float, default=1.0)
    parser.add_argument("--top-logit-atol", type=float, default=1.0)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")
    result = run_real_traceable_decode_logits_runner(
        args.snapshot_dir,
        layer=args.layer,
        prefill_seq_len=args.prefill_seq_len,
        decode_steps=args.decode_steps,
        input_ids=args.input_ids,
        input_ids_path=args.input_ids_path,
        prompt_path=args.prompt_path,
        input_id_start=args.input_id_start,
        prompt_label=args.prompt_label,
        embedding_mode=args.embedding_mode,  # type: ignore[arg-type]
        max_embedding_rows=args.max_embedding_rows,
        vocab_mode=args.vocab_mode,  # type: ignore[arg-type]
        full_vocab_smoke=args.full_vocab_smoke,
        vocab_start=args.vocab_start,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        trace_region_size=args.trace_region_size,
        cache_len=args.cache_len,
        attention_mode=args.attention_mode,
        attention_read_api=args.attention_read_api,
        cache_update_api=args.cache_update_api,
        rope_position_api=args.rope_position_api,
        sparse_indexer_mode=args.sparse_indexer_mode,
        compressor_mode=args.compressor_mode,
        indexer_compressor_mode=args.indexer_compressor_mode,
        routed_topk_prefix=args.routed_topk_prefix,
        trace_pcc=args.trace_pcc,
        final_norm_pcc=args.final_norm_pcc,
        logits_pcc=args.logits_pcc,
        rtol=args.rtol,
        atol=args.atol,
        logits_rtol=args.logits_rtol,
        logits_atol=args.logits_atol,
        top_logit_atol=args.top_logit_atol,
    )
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _validate_runner_args(
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    prefill_seq_len: int,
    decode_steps: int,
    cache_len: int,
    max_tensors: int,
    max_bytes: int,
) -> None:
    if not 0 <= int(layer) < int(config.num_hidden_layers):
        raise ValueError(f"layer must be in [0, {config.num_hidden_layers}), got {layer}")
    if int(prefill_seq_len) < 2:
        raise ValueError(f"prefill_seq_len must be at least 2 for the trace envelope, got {prefill_seq_len}")
    if int(decode_steps) <= 0:
        raise ValueError(f"decode_steps must be positive, got {decode_steps}")
    if int(prefill_seq_len) + int(decode_steps) > int(cache_len):
        raise ValueError(
            f"cache_len must cover positions through {int(prefill_seq_len) + int(decode_steps) - 1}, got {cache_len}"
        )
    if int(max_tensors) <= 0 or int(max_bytes) <= 0:
        raise ValueError("max_tensors and max_bytes must be positive")


def _build_real_embedding_trace_activations(
    hidden_states: torch.Tensor,
    *,
    prefill_seq_len: int,
    decode_steps: int,
) -> tuple[torch.Tensor, ...]:
    if hidden_states.ndim != 4 or tuple(hidden_states.shape[:2]) != (1, 1):
        raise ValueError(f"hidden_states must have shape [1, 1, tokens, hidden], got {tuple(hidden_states.shape)}")
    activations = []
    context_rows = int(prefill_seq_len) - 1
    for step in range(int(decode_steps)):
        decode_index = int(prefill_seq_len) + step
        decode_row = hidden_states[:, :, decode_index : decode_index + 1, :]
        context_end = decode_index
        context_start = context_end - context_rows
        if context_start < 0:
            raise ValueError("not enough embedded context rows to build traceable decode activation")
        context = hidden_states[:, :, context_start:context_end, :]
        activation = torch.cat([decode_row, context], dim=2).contiguous().to(torch.bfloat16)
        activations.append(activation)
    return tuple(activations)


def _run_logits_steps(
    reference_steps: Sequence[Mapping[str, torch.Tensor]],
    *,
    ttnn_steps: Sequence[Mapping[str, torch.Tensor]] | None,
    logits_weights,
    config: DeepSeekV4FlashConfig,
    top_k: int,
    cpu_only: bool,
    device_id: int,
    final_norm_pcc: float,
    logits_pcc: float,
    rtol: float,
    atol: float,
    logits_rtol: float,
    logits_atol: float,
    top_logit_atol: float,
) -> dict[str, Any]:
    reference_logits = []
    ttnn_logits = []
    step_results = []
    for step, reference in enumerate(reference_steps):
        reference_hidden = _logical_decode_hidden(reference["residual_output"])
        reference_logit = build_torch_decode_logits_reference(
            reference_hidden,
            logits_weights.norm_weight,
            logits_weights.head_weight,
            config=config,
        )
        reference_logits.append(reference_logit)
        step_result = {
            "step_index": int(step),
            "reference": {
                "final_norm": _tensor_summary(reference_logit["final_norm"]),
                "logits": _tensor_summary(reference_logit["logits"]),
                "top_k": _topk_summary(reference_logit["logits"], top_k=top_k, vocab_start=logits_weights.vocab_start),
            },
            "ttnn": {},
            "accuracy": {},
            "passed": bool(cpu_only),
        }
        if not cpu_only:
            if ttnn_steps is None:
                raise ValueError("TTNN trace outputs are required unless cpu_only=True")
            ttnn_hidden = _logical_decode_hidden(ttnn_steps[step]["residual_output"])
            ttnn_logit = _run_ttnn_decode_logits_from_hidden(
                ttnn_hidden,
                logits_weights.norm_weight,
                logits_weights.head_weight,
                config=config,
                device_id=device_id,
            )
            ttnn_logits.append(ttnn_logit)
            top_k_accuracy = _topk_accuracy_summary(
                reference_logit["logits"],
                ttnn_logit["logits"],
                top_k=top_k,
                vocab_start=logits_weights.vocab_start,
                top_logit_atol=top_logit_atol,
            )
            step_result["ttnn"] = {
                "final_norm": _tensor_summary(ttnn_logit["final_norm"]),
                "logits": _tensor_summary(ttnn_logit["logits"]),
                "top_k": _topk_summary(ttnn_logit["logits"], top_k=top_k, vocab_start=logits_weights.vocab_start),
            }
            step_result["accuracy"] = {
                "final_norm": _accuracy_summary(
                    reference_logit["final_norm"],
                    ttnn_logit["final_norm"],
                    pcc_threshold=final_norm_pcc,
                    rtol=rtol,
                    atol=atol,
                ),
                "logits": _dense_pcc_summary(
                    reference_logit["logits"],
                    ttnn_logit["logits"],
                    pcc_threshold=logits_pcc,
                    rtol=logits_rtol,
                    atol=logits_atol,
                ),
                "top_k": {
                    **top_k_accuracy,
                    "required_for_pass": logits_weights.vocab_mode == "full",
                    "pass_policy": "diagnostic_only_for_sliced_vocab"
                    if logits_weights.vocab_mode == "slice"
                    else "required_for_full_vocab",
                },
            }
            required_accuracy = [
                item for item in step_result["accuracy"].values() if bool(item.get("required_for_pass", True))
            ]
            step_result["passed"] = all(item["passed"] for item in required_accuracy)
        step_results.append(step_result)
    return {
        "mode": "torch-reference-only" if cpu_only else "ttnn",
        "steps": step_results,
        "passed": all(step["passed"] for step in step_results),
        "ttnn_compared_to_torch": not cpu_only,
        "reference_logits": reference_logits,
        "ttnn_logits": ttnn_logits,
    }


def _build_result(
    *,
    snapshot_dir: Path,
    config: DeepSeekV4FlashConfig,
    trace_result: Mapping[str, Any],
    logits_result: Mapping[str, Any],
    embedding: InputEmbeddingPayload,
    logits_weights,
    input_ids: torch.Tensor,
    resolved_input,
    full_vocab_smoke: bool,
    max_tensors: int,
    max_bytes: int,
    wall_seconds: float,
) -> dict[str, Any]:
    positions = [int(value) for value in trace_result["positions"]]
    logits_steps = _public_logits_steps(logits_result["steps"], positions=positions, input_ids=input_ids)
    protected_steps = _protected_steps(trace_result, positions=positions, config=config)
    host_outside = [
        *trace_result["host_boundaries_outside_trace"],
        "embedding_weight_load",
        "embedding_lookup_host",
        "final_norm_lm_head_weight_load",
        "final_norm_lm_head_execution_outside_protected_layer",
        "logits_topk_and_accuracy_readback",
    ]
    ttnn_logits_ops = (
        ["ttnn.rms_norm(final_norm_outside_protected_layer)", f"ttnn.linear(lm_head_{logits_weights.vocab_mode})"]
        if logits_result["mode"] == "ttnn"
        else []
    )
    return {
        "schema_version": REAL_TRACEABLE_DECODE_LOGITS_RUNNER_SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "mode": trace_result["mode"],
        "passed": bool(trace_result["passed"] and logits_result["passed"]),
        "snapshot_dir": str(snapshot_dir),
        "layers": [int(trace_result["layer"])],
        "prefill_sequence_length": int(trace_result["tensor_sequence_length"]),
        "decode_steps": int(trace_result["decode_step_count"]),
        "decode_positions": positions,
        "input": {
            "source": resolved_input.source,
            "source_path": resolved_input.source_path,
            "prompt_label": resolved_input.prompt_label,
            "token_ids": [int(value) for value in input_ids[0].tolist()],
            "prefill_token_ids": [
                int(value) for value in input_ids[0, : int(trace_result["tensor_sequence_length"])].tolist()
            ],
            "supplied_decode_token_ids": [
                int(value) for value in input_ids[0, int(trace_result["tensor_sequence_length"]) :].tolist()
            ],
            "activation_source": trace_result["activation_source"],
            "trace_envelope": (
                "row 0 is the supplied decode-token embedding; remaining rows are real embedded context tokens"
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
            "payload_bytes": {"embedding": int(embedding.metadata.nbytes), "total": int(embedding.metadata.nbytes)},
        },
        "protected_decode": {
            "passed": bool(trace_result["passed"]),
            "mode": trace_result["mode"],
            "layer": int(trace_result["layer"]),
            "attention_read_api": trace_result["attention_read_api"],
            "sparse_indexer_mode": trace_result["sparse_indexer_mode"],
            "compressor_mode": trace_result["compressor_mode"],
            "indexer_compressor_mode": trace_result["indexer_compressor_mode"],
            "trace_capture": {
                "attempted": trace_result["trace_capture_attempted"],
                "capture_passed": trace_result["trace_capture_passed"],
                "execute_replay_passed": trace_result["trace_execute_replay_passed"],
                "capture_count": trace_result["trace_capture"]["capture_count"],
                "replay_count": trace_result["trace_capture"]["decode_step_count"],
                "recaptured_per_position": trace_result["multi_position_replay"]["recaptured_per_position"],
                "single_capture_replayed_across_positions": trace_result["one_trace_capture_replayed_across_positions"],
                "per_step_trace_variants": trace_result.get("per_step_trace_variants", []),
            },
            "carried_device_state": {
                "kv_cache": trace_result["trace_capture"].get("carried_device_kv_cache_state"),
                "sparse_kv_cache": trace_result["trace_capture"].get("carried_device_sparse_kv_cache_state"),
                "compressor": trace_result["trace_capture"].get("carried_device_compressor_state"),
                "indexer_compressor": trace_result["trace_capture"].get("carried_device_indexer_compressor_state"),
                "rebuilt_on_host_between_replay_steps": trace_result["trace_capture"].get(
                    "state_rebuilt_on_host_between_replay_steps"
                ),
            },
            "guard_status": trace_result["guard_status"],
            "host_boundaries_inside_trace": trace_result["host_boundaries_inside_trace"],
            "selected_rows": _selected_rows_report(trace_result, config=config),
            "accuracy": trace_result.get("per_step_accuracy_summary", []),
            "steps": protected_steps,
        },
        "logits": {
            "mode": logits_result["mode"],
            "ttnn_compared_to_torch": logits_result["ttnn_compared_to_torch"],
            "vocab": {
                "mode": logits_weights.vocab_mode,
                "is_sliced": logits_weights.vocab_mode == "slice",
                "vocab_start": logits_weights.vocab_start,
                "vocab_size": logits_weights.vocab_size,
                "full_vocab_size": logits_weights.full_vocab_size,
                "full_vocab_smoke_requested": bool(full_vocab_smoke),
                "deterministic_slice": (
                    None
                    if logits_weights.vocab_mode == "full"
                    else f"[{logits_weights.vocab_start}, {logits_weights.vocab_start + logits_weights.vocab_size})"
                ),
            },
            "final_norm_lm_head": {
                "loaded_keys": {"final_norm": "norm.weight", "lm_head": logits_weights.head_key},
                "loaded_source_keys": {"final_norm": "norm.weight", "lm_head": logits_weights.head_source},
                "lm_head_shape_loaded": list(logits_weights.head_weight.shape),
                "lm_head_full_shape": [logits_weights.full_vocab_size, int(config.hidden_size)],
            },
            "steps": logits_steps,
            "passed": bool(logits_result["passed"]),
        },
        "ttnn_ops": [*trace_result.get("ttnn_ops", []), *ttnn_logits_ops],
        "host_boundaries": {
            "inside_protected_execution": trace_result["host_boundaries_inside_trace"],
            "outside_protected_execution": host_outside,
        },
        "payload_bytes": {
            "embedding": int(embedding.metadata.nbytes),
            "protected_decode": int(trace_result["payload_bytes"]["total"]),
            "final_norm_lm_head": sum(item.nbytes for item in logits_weights.metadata),
            "total": int(embedding.metadata.nbytes)
            + int(trace_result["payload_bytes"]["total"])
            + sum(item.nbytes for item in logits_weights.metadata),
        },
        "loaded_tensors": [
            _metadata_summary(embedding.metadata),
            *trace_result["loaded_tensors"],
            *[_metadata_summary(item) for item in logits_weights.metadata],
        ],
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
        },
        "model": trace_result["model"],
        "timing": {"runner_wall_seconds": wall_seconds},
        "limitations": _limitations(trace_result),
    }


def _protected_steps(
    trace_result: Mapping[str, Any],
    *,
    positions: Sequence[int],
    config: DeepSeekV4FlashConfig,
) -> list[dict[str, Any]]:
    accuracy_by_step = {int(item["step"]): item["accuracy"] for item in trace_result.get("accuracy_by_step", [])}
    variants = {int(item["step"]): item for item in trace_result.get("per_step_trace_variants", [])}
    rows_by_step = {int(item["step"]): item for item in _selected_rows_report(trace_result, config=config)}
    steps = []
    for step, detail in enumerate(trace_result["decode_steps_detail"]):
        rows = rows_by_step.get(step, {})
        steps.append(
            {
                "step_index": int(step),
                "position": int(positions[step]),
                "trace_variant": variants.get(step),
                "selected_cache_rows": detail.get("selected_cache_rows"),
                "selected_compressed_rows": rows.get("compressed_rows", []),
                "accuracy": _compact_protected_accuracy(accuracy_by_step.get(step, {})),
            }
        )
    return steps


def _public_logits_steps(
    logits_steps: Sequence[Mapping[str, Any]],
    *,
    positions: Sequence[int],
    input_ids: torch.Tensor,
) -> list[dict[str, Any]]:
    prefill_seq_len = int(positions[0])
    public = []
    for step, item in enumerate(logits_steps):
        public.append(
            {
                "step_index": int(step),
                "position": int(positions[step]),
                "feed_token_id": int(input_ids[0, prefill_seq_len + step].item()),
                "reference": item["reference"],
                "ttnn": item["ttnn"],
                "accuracy": item["accuracy"],
                "passed": item["passed"],
            }
        )
    return public


def _compact_protected_accuracy(accuracy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attention": _accuracy_items(accuracy, ("attention_output", "attention_projected", "residual_output")),
        "router": _accuracy_items(accuracy, ("router_decode_topk_indices", "router_decode_route_weights")),
        "ffn": _accuracy_items(accuracy, ("shared_output", "routed_output", "combined_ffn_output", "residual_output")),
    }


def _accuracy_items(accuracy: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    items = {name: accuracy[name] for name in names if name in accuracy}
    pcc_values = [float(item["pcc"]) for item in items.values() if item.get("pcc") is not None]
    return {
        "passed": all(bool(item.get("passed", False)) for item in items.values()) if items else None,
        "min_pcc": min(pcc_values) if pcc_values else None,
        "items": items,
    }


def _selected_rows_report(
    trace_result: Mapping[str, Any], *, config: DeepSeekV4FlashConfig | None = None
) -> list[dict[str, Any]]:
    if trace_result.get("device_selected_cache_rows_by_step"):
        return list(trace_result["device_selected_cache_rows_by_step"])
    sliding_window = int(config.sliding_window) if config is not None else 0
    report = []
    for detail in trace_result.get("decode_steps_detail", []):
        rows = [int(value) for value in (detail.get("selected_cache_rows") or [])]
        compressed_rows = [value for value in rows if value >= sliding_window]
        report.append(
            {
                "step": int(detail["step"]),
                "position": int(detail["position"]),
                "rows": rows,
                "static_window_rows": [value for value in rows if value < sliding_window],
                "compressed_rows": compressed_rows,
                "compressed_indices": [value - sliding_window for value in compressed_rows],
                "rows_source": detail.get("selected_row_ids_source"),
            }
        )
    return report


def _logical_decode_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 4 or int(hidden_states.shape[-2]) < 1:
        raise ValueError(f"expected hidden_states [batch, 1, tokens, hidden], got {tuple(hidden_states.shape)}")
    return hidden_states[:, :, :1, :].contiguous()


def _limitations(trace_result: Mapping[str, Any]) -> list[str]:
    return [
        "only one protected decoder layer is run; no two-layer/full-model device stack is claimed",
        "final RMSNorm and LM head run after protected decode output readback, outside the protected layer region",
        "embedding lookup and trace-envelope assembly are host-side input glue",
        "the trace envelope uses real token embeddings but is not a full prefill-to-decode hidden-state handoff",
        *[str(item) for item in trace_result.get("remaining_limitations", [])],
    ]


def _seconds_since(start: float) -> float:
    return round(max(0.0, time.perf_counter() - start), 6)


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()
