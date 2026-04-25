# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    RealCheckpointTensorIndex,
    TensorMetadata,
    _safetensors_nbytes,
)
from models.demos.deepseek_v4_flash.real_decode_logits_smoke import DEFAULT_DECODE_LOGITS_TOP_K, VocabMode
from models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke import (
    DEFAULT_DECODE_STACK_LAYERS,
    DEFAULT_DECODE_STACK_MAX_BYTES,
    DEFAULT_DECODE_STACK_MAX_TENSORS,
    run_real_decode_stack_logits_smoke,
)
from models.demos.deepseek_v4_flash.real_ffn_smoke import _metadata_summary, _tensor_summary
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import DEFAULT_SEQUENCE_LENGTH

REAL_INPUT_STACK_LOGITS_SMOKE_SCHEMA_VERSION = 1
DEFAULT_INPUT_STACK_MAX_TENSORS = DEFAULT_DECODE_STACK_MAX_TENSORS + 1
DEFAULT_INPUT_STACK_MAX_BYTES = DEFAULT_DECODE_STACK_MAX_BYTES + 1024 * 1024 * 1024
DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS = 1024
EmbeddingMode = Literal["slice", "full"]


@dataclass(frozen=True)
class InputEmbeddingPayload:
    hidden_states: torch.Tensor
    metadata: TensorMetadata
    mode: EmbeddingMode
    row_start: int
    row_end: int
    full_vocab_size: int
    hidden_size: int


def run_real_input_stack_logits_smoke(
    snapshot_dir: str | Path,
    *,
    layers: Sequence[int] = DEFAULT_DECODE_STACK_LAYERS,
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    input_ids: Sequence[int] | torch.Tensor | None = None,
    input_id_start: int = 0,
    prompt_label: str | None = None,
    embedding_mode: EmbeddingMode = "slice",
    max_embedding_rows: int = DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS,
    vocab_mode: VocabMode = "slice",
    vocab_start: int = 0,
    vocab_size: int | None = None,
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K,
    max_tensors: int = DEFAULT_INPUT_STACK_MAX_TENSORS,
    max_bytes: int = DEFAULT_INPUT_STACK_MAX_BYTES,
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
    """Run real token IDs through embedding, layers 2->3, final norm, and LM head."""

    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    input_source = "deterministic_contiguous_input_ids" if input_ids is None else "explicit_input_ids"
    token_ids = _normalize_input_ids(
        input_ids,
        prefill_seq_len=prefill_seq_len,
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

    initial_prefill_hidden = embedding.hidden_states[:, :, :prefill_seq_len, :].contiguous()
    initial_decode_hidden = embedding.hidden_states[:, :, prefill_seq_len:, :].contiguous()
    result = run_real_decode_stack_logits_smoke(
        snapshot_dir,
        layers=layers,
        prefill_seq_len=prefill_seq_len,
        vocab_mode=vocab_mode,
        vocab_start=vocab_start,
        vocab_size=vocab_size,
        top_k=top_k,
        max_tensors=remaining_tensors,
        max_bytes=remaining_bytes,
        cpu_only=cpu_only,
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
        initial_prefill_hidden=initial_prefill_hidden,
        initial_decode_hidden=initial_decode_hidden,
        input_ids=token_ids,
    )
    _augment_result_for_real_input_ids(
        result,
        embedding=embedding,
        input_ids=token_ids,
        prompt_label=prompt_label,
        input_source=input_source,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    return result


def load_input_embedding_payload(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    input_ids: torch.Tensor,
    embedding_mode: EmbeddingMode,
    max_embedding_rows: int,
) -> InputEmbeddingPayload:
    if embedding_mode not in ("slice", "full"):
        raise ValueError(f"embedding_mode must be 'slice' or 'full', got {embedding_mode!r}")
    if max_embedding_rows <= 0:
        raise ValueError(f"max_embedding_rows must be positive, got {max_embedding_rows}")

    location = index.location("embed.weight")
    full_metadata = index.metadata_for_keys(["embed.weight"])[0]
    _validate_embedding_metadata(full_metadata, config=config)

    if embedding_mode == "full":
        row_start = 0
        row_end = int(config.vocab_size)
        metadata = full_metadata
    else:
        row_start = int(input_ids.min().item())
        row_end = int(input_ids.max().item()) + 1
        row_count = row_end - row_start
        if row_count > max_embedding_rows:
            raise ValueError(
                f"embedding slice spans {row_count} rows, exceeding max_embedding_rows={max_embedding_rows}; "
                "use --embedding-mode full or choose a tighter deterministic input-id range"
            )
        metadata = TensorMetadata(
            canonical_key=full_metadata.canonical_key,
            source_key=full_metadata.source_key,
            shard_name=full_metadata.shard_name,
            shard_path=full_metadata.shard_path,
            dtype=full_metadata.dtype,
            shape=(row_count, int(config.hidden_size)),
            nbytes=_safetensors_nbytes(full_metadata.dtype, (row_count, int(config.hidden_size))),
        )

    with safe_open(location.shard_path, framework="pt", device="cpu") as handle:
        if embedding_mode == "full":
            embedding_rows = handle.get_tensor(location.source_key).contiguous()
            local_ids = input_ids.to(torch.long)
        else:
            embedding_rows = handle.get_slice(location.source_key)[row_start:row_end].contiguous()
            local_ids = input_ids.to(torch.long) - row_start

    hidden_states = F.embedding(local_ids, embedding_rows.to(torch.bfloat16)).unsqueeze(1).contiguous()
    return InputEmbeddingPayload(
        hidden_states=hidden_states,
        metadata=metadata,
        mode=embedding_mode,
        row_start=row_start,
        row_end=row_end,
        full_vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real DeepSeek V4 Flash input IDs through embedding, a two-layer decode stack, "
            "final norm, and LM head."
        )
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_DECODE_STACK_LAYERS))
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--input-id-start", type=int, default=0)
    parser.add_argument("--prompt-label")
    parser.add_argument("--embedding-mode", choices=("slice", "full"), default="slice")
    parser.add_argument("--max-embedding-rows", type=int, default=DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS)
    parser.add_argument(
        "--vocab-mode",
        choices=("full", "slice"),
        default=os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_LOGITS_VOCAB_MODE", "slice"),
    )
    parser.add_argument(
        "--vocab-start",
        type=int,
        default=_optional_int_env(
            "DSV4_FLASH_REAL_INPUT_STACK_LOGITS_VOCAB_START",
            _optional_int_env("DSV4_FLASH_REAL_INPUT_STACK_VOCAB_START", 0),
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_optional_int_env(
            "DSV4_FLASH_REAL_INPUT_STACK_LOGITS_VOCAB_SIZE",
            _optional_int_env("DSV4_FLASH_REAL_INPUT_STACK_VOCAB_SIZE"),
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_INPUT_STACK_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_INPUT_STACK_MAX_BYTES)
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

    result = run_real_input_stack_logits_smoke(
        args.snapshot_dir,
        layers=args.layers,
        prefill_seq_len=args.prefill_seq_len,
        input_ids=args.input_ids,
        input_id_start=args.input_id_start,
        prompt_label=args.prompt_label,
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
    result.pop("_reference_tensors", None)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _normalize_input_ids(
    input_ids: Sequence[int] | torch.Tensor | None,
    *,
    prefill_seq_len: int,
    vocab_size: int,
    input_id_start: int,
) -> torch.Tensor:
    if prefill_seq_len <= 0:
        raise ValueError(f"prefill_seq_len must be positive, got {prefill_seq_len}")
    total_tokens = int(prefill_seq_len) + 1
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


def _validate_embedding_metadata(metadata: TensorMetadata, *, config: DeepSeekV4FlashConfig) -> None:
    expected = (int(config.vocab_size), int(config.hidden_size))
    if tuple(metadata.shape) != expected:
        raise ValueError(f"Expected embed.weight shape {expected}, got {metadata.shape}")
    if metadata.dtype not in {"BF16", "F16", "F32"}:
        raise ValueError(f"Unsupported embedding dtype {metadata.dtype}")


def _augment_result_for_real_input_ids(
    result: dict[str, Any],
    *,
    embedding: InputEmbeddingPayload,
    input_ids: torch.Tensor,
    prompt_label: str | None,
    input_source: str,
    max_tensors: int,
    max_bytes: int,
) -> None:
    prefill_seq_len = int(result["prefill_sequence_length"])
    embedding_payload = {
        "embedding": int(embedding.metadata.nbytes),
        "total": int(embedding.metadata.nbytes),
    }
    result["schema_version"] = REAL_INPUT_STACK_LOGITS_SMOKE_SCHEMA_VERSION
    result["activation_source"] = "real_input_ids_host_embedding_lookup"
    result["input"] = {
        "source": input_source,
        "prompt_label": prompt_label,
        "token_ids": [int(value) for value in input_ids[0].tolist()],
        "prefill_token_ids": [int(value) for value in input_ids[0, :prefill_seq_len].tolist()],
        "decode_token_ids": [int(value) for value in input_ids[0, prefill_seq_len:].tolist()],
        "prefill_tokens": prefill_seq_len,
        "decode_tokens": 1,
    }
    result["embedding"] = {
        "mode": embedding.mode,
        "loaded_key": "embed.weight",
        "loaded_source_key": embedding.metadata.source_key,
        "loaded_shape": list(embedding.metadata.shape),
        "full_shape": [embedding.full_vocab_size, embedding.hidden_size],
        "row_start": int(embedding.row_start),
        "row_end": int(embedding.row_end),
        "deterministic_slice": (None if embedding.mode == "full" else f"[{embedding.row_start}, {embedding.row_end})"),
        "payload_bytes": embedding_payload,
        "embedded_hidden_states": _tensor_summary(embedding.hidden_states),
    }
    result["stack_scope"]["path"] = (
        "real input_ids -> host embedding lookup -> layer 2 prefill cache and one-token decode -> "
        "layer 3 prefill cache from layer 2 prefill output and one-token decode -> final norm -> LM head"
    )
    result["stack_scope"][
        "embeddings_vllm_evals"
    ] = "real token embedding lookup included; tokenizer/vLLM/full serving remains excluded"
    result["loaded_tensors"].append(_metadata_summary(embedding.metadata))
    result["loaded_tensor_groups"]["embedding"] = {
        "count": 1,
        "payload_bytes": embedding_payload,
        "canonical_keys": [embedding.metadata.canonical_key],
    }
    result["payload_bytes"]["embedding"] = embedding_payload
    result["payload_bytes"]["total"] = int(result["payload_bytes"]["total"]) + int(embedding.metadata.nbytes)
    result["budget"]["max_tensors"] = int(max_tensors)
    result["budget"]["max_bytes"] = int(max_bytes)
    result["budget"]["selected_tensors"] = int(result["budget"]["selected_tensors"]) + 1
    result["budget"]["selected_payload_bytes"] = int(result["budget"]["selected_payload_bytes"]) + int(
        embedding.metadata.nbytes
    )
    result["host_boundaries"].insert(
        0,
        {
            "name": "embedding_weight_load",
            "location": "checkpoint load",
            "description": "real token embedding rows are loaded on host before the stack smoke",
        },
    )
    result["host_boundaries"].insert(
        1,
        {
            "name": "embedding_lookup_host",
            "location": "before first decoder layer",
            "description": "input_ids are embedded on host to feed the existing real layer stack boundary",
        },
    )
    result["reference_ops"].insert(0, "torch.embedding(real_embed_weight)")


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()
