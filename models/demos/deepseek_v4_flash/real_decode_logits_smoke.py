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
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    RealCheckpointTensorIndex,
    TensorMetadata,
    _safetensors_nbytes,
)
from models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke import (
    DEFAULT_DECODE_DECODER_LAYER,
    DEFAULT_DECODE_DECODER_LAYER_EXPERT,
    DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES,
    DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS,
)
from models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke import _base_result as _decode_base_result
from models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke import (
    _prepare_decode_ffn_fanout,
    _run_ttnn_decode_decoder_layer_slice,
    _validate_runtime_config,
    _validate_smoke_args,
    build_torch_decode_attention_runtime_reference,
    build_torch_decode_cache_prep_reference,
)
from models.demos.deepseek_v4_flash.real_ffn_smoke import _accuracy_summary, _pcc, _tensor_summary
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import decode_real_kv_projection_weights
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE,
    _query_projection_weights_only,
    decode_real_prefill_attention_projection_weights,
    layer_prefill_attention_runtime_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    DEFAULT_SEQUENCE_LENGTH,
    build_torch_prefill_cache_prep_reference,
)
from models.demos.deepseek_v4_flash.real_prefill_decoder_layer_smoke import (
    _load_attention_and_ffn_selector_slice,
    _metadata_groups,
    _residual_add,
    _unique_keys,
)

REAL_DECODE_LOGITS_SMOKE_SCHEMA_VERSION = 1
DEFAULT_DECODE_LOGITS_LAYER = DEFAULT_DECODE_DECODER_LAYER
DEFAULT_DECODE_LOGITS_EXPERT = DEFAULT_DECODE_DECODER_LAYER_EXPERT
DEFAULT_DECODE_LOGITS_MAX_TENSORS = DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS + 2
DEFAULT_DECODE_LOGITS_MAX_BYTES = DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES + 2 * 1024 * 1024 * 1024
DEFAULT_DECODE_LOGITS_TOP_K = 5
DEFAULT_DECODE_LOGITS_VOCAB_SLICE_SIZE = 4096
VocabMode = Literal["full", "slice"]


@dataclass(frozen=True)
class DecodeLogitsWeights:
    norm_weight: torch.Tensor
    head_weight: torch.Tensor
    metadata: tuple[TensorMetadata, ...]
    head_key: str
    head_source: str
    vocab_mode: VocabMode
    vocab_start: int
    vocab_size: int
    full_vocab_size: int


def run_real_decode_logits_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_DECODE_LOGITS_LAYER,
    expert: int | None = DEFAULT_DECODE_LOGITS_EXPERT,
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    vocab_mode: VocabMode = "full",
    vocab_start: int = 0,
    vocab_size: int | None = None,
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K,
    max_tensors: int = DEFAULT_DECODE_LOGITS_MAX_TENSORS,
    max_bytes: int = DEFAULT_DECODE_LOGITS_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    decoder_final_pcc: float = 0.99,
    final_norm_pcc: float = 0.999,
    logits_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    decoder_residual_atol: float = 3e-1,
    logits_rtol: float = 1e-1,
    logits_atol: float = 1.0,
    top_logit_atol: float = 1.0,
) -> dict[str, Any]:
    """Run a real one-layer decode hidden state through final norm and LM head."""

    _validate_smoke_args(
        layer=layer,
        expert=expert,
        prefill_seq_len=prefill_seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        pcc_thresholds={
            "decoder_final_pcc": decoder_final_pcc,
            "final_norm_pcc": final_norm_pcc,
            "logits_pcc": logits_pcc,
        },
    )
    _validate_logits_args(vocab_mode=vocab_mode, vocab_start=vocab_start, vocab_size=vocab_size, top_k=top_k)
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    _validate_runtime_config(config, layer=layer, prefill_seq_len=prefill_seq_len)
    current_position = int(prefill_seq_len)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)

    tensors, metadata, attention_keys, selector_keys = _load_attention_and_ffn_selector_slice(
        index,
        config=config,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    q_weights = decode_real_prefill_attention_projection_weights(tensors, config=config, layer=layer)
    kv_weights = decode_real_kv_projection_weights(tensors, config=config, layer=layer)

    from models.demos.deepseek_v4_flash.real_attention_projection_smoke import deterministic_attention_activation

    prefill_activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=prefill_seq_len)
    decode_activation = deterministic_attention_activation(
        hidden_size=config.hidden_size,
        seq_len=prefill_seq_len + 1,
    )[:, :, -1:, :].contiguous()
    prefill_cache_reference = build_torch_prefill_cache_prep_reference(
        tensors,
        _query_projection_weights_only(q_weights),
        kv_weights,
        config=config,
        layer=layer,
        activation=prefill_activation,
        start_pos=0,
    )
    decode_cache_reference = build_torch_decode_cache_prep_reference(
        tensors,
        _query_projection_weights_only(q_weights),
        kv_weights,
        config=config,
        layer=layer,
        activation=decode_activation,
        current_position=current_position,
    )
    attention_reference = build_torch_decode_attention_runtime_reference(
        prefill_cache_reference,
        decode_cache_reference,
        q_weights,
        tensors[f"layers.{layer}.attn.attn_sink"],
        config=config,
        layer=layer,
        current_position=current_position,
    )
    post_attention_residual = _residual_add(decode_activation, attention_reference["attention_output_projected"])

    (
        tensors,
        metadata,
        ffn_keys,
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
        layer=layer,
        requested_expert=expert,
        activation=post_attention_residual,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
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
    logits_reference = build_torch_decode_logits_reference(
        ffn_reference["residual_output"],
        logits_weights.norm_weight,
        logits_weights.head_weight,
        config=config,
    )
    reference = {
        "prefill_input_hidden_states": prefill_activation,
        "decode_input_hidden_states": decode_activation,
        "prefill_cache": prefill_cache_reference,
        "decode_cache": decode_cache_reference,
        "attention": attention_reference,
        "post_attention_residual": post_attention_residual,
        "ffn": ffn_reference,
        "post_ffn_residual": ffn_reference["residual_output"],
        "final_norm": logits_reference["final_norm"],
        "logits": logits_reference["logits"],
    }
    all_metadata = [*metadata, *logits_weights.metadata]
    metadata_groups = _metadata_groups(
        all_metadata,
        attention_keys=attention_keys,
        selector_keys=selector_keys,
        ffn_keys=ffn_keys,
    )
    metadata_groups["final_norm_lm_head"] = list(logits_weights.metadata)
    result = _decode_base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        activated_experts=activated_experts,
        requested_expert=expert,
        prefill_seq_len=prefill_seq_len,
        current_position=current_position,
        config=config,
        metadata=all_metadata,
        metadata_groups=metadata_groups,
        reference=reference,
        router_preview=router_preview,
        ffn_input_ids=ffn_input_ids,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    _augment_base_result_for_logits(
        result,
        config=config,
        logits_weights=logits_weights,
        reference=reference,
        top_k=top_k,
    )

    if cpu_only:
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
            f"TTNN prefill cache build seq_len must be a multiple of "
            f"{PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE}, got {prefill_seq_len}"
        )

    ttnn_decoder_outputs = _run_ttnn_decode_decoder_layer_slice(
        tensors,
        q_weights,
        kv_weights,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=layer,
        input_ids=ffn_input_ids,
        prefill_activation=prefill_activation,
        decode_activation=decode_activation,
        current_position=current_position,
        device_id=device_id,
    )
    ttnn_logits_outputs = _run_ttnn_decode_logits_from_hidden(
        ttnn_decoder_outputs["post_ffn_residual"],
        logits_weights.norm_weight,
        logits_weights.head_weight,
        config=config,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        *result["ttnn_ops"],
        "ttnn.rms_norm(final_norm)",
        f"ttnn.linear(lm_head_{logits_weights.vocab_mode})",
    ]
    result["ttnn"]["final_norm"] = _tensor_summary(ttnn_logits_outputs["final_norm"])
    result["ttnn"]["logits"] = _tensor_summary(ttnn_logits_outputs["logits"])
    result["ttnn"]["top_k"] = _topk_summary(
        ttnn_logits_outputs["logits"],
        top_k=top_k,
        vocab_start=logits_weights.vocab_start,
    )
    result["accuracy"] = {
        "decoder_post_ffn_residual": _accuracy_summary(
            reference["post_ffn_residual"],
            ttnn_decoder_outputs["post_ffn_residual"],
            pcc_threshold=decoder_final_pcc,
            rtol=rtol,
            atol=decoder_residual_atol,
        ),
        "final_norm": _accuracy_summary(
            reference["final_norm"],
            ttnn_logits_outputs["final_norm"],
            pcc_threshold=final_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "logits": _dense_pcc_summary(
            reference["logits"],
            ttnn_logits_outputs["logits"],
            pcc_threshold=logits_pcc,
            rtol=logits_rtol,
            atol=logits_atol,
        ),
        "top_k": _topk_accuracy_summary(
            reference["logits"],
            ttnn_logits_outputs["logits"],
            top_k=top_k,
            vocab_start=logits_weights.vocab_start,
            top_logit_atol=top_logit_atol,
        ),
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_decode_logits_keys(
    index: RealCheckpointTensorIndex, *, config: DeepSeekV4FlashConfig, layer: int
) -> list[str]:
    return _unique_keys(
        [
            *layer_prefill_attention_runtime_keys(index, config=config, layer=layer),
            "norm.weight",
            _resolve_head_key(index),
        ]
    )


def load_decode_logits_weights(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    vocab_mode: VocabMode,
    vocab_start: int,
    vocab_size: int | None,
    already_loaded_metadata: Sequence[TensorMetadata],
    max_tensors: int,
    max_bytes: int,
) -> DecodeLogitsWeights:
    head_key = _resolve_head_key(index)
    full_metadata = index.metadata_for_keys(["norm.weight", head_key])
    metadata = tuple(
        _slice_head_metadata(item, config=config, vocab_mode=vocab_mode, vocab_start=vocab_start, vocab_size=vocab_size)
        for item in full_metadata
    )
    _enforce_total_budget(
        [*already_loaded_metadata, *metadata],
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    norm_metadata = _metadata_for_key(metadata, "norm.weight")
    head_metadata = _metadata_for_key(metadata, head_key)
    _validate_logits_metadata(norm_metadata, head_metadata, config=config, vocab_mode=vocab_mode)

    norm_location = index.location("norm.weight")
    with safe_open(norm_location.shard_path, framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor(norm_location.source_key).contiguous()

    head_location = index.location(head_key)
    with safe_open(head_location.shard_path, framework="pt", device="cpu") as handle:
        if vocab_mode == "full":
            head_weight = handle.get_tensor(head_location.source_key).contiguous()
        else:
            end = head_metadata.shape[0] + vocab_start
            head_weight = handle.get_slice(head_location.source_key)[vocab_start:end].contiguous()

    return DecodeLogitsWeights(
        norm_weight=norm_weight,
        head_weight=head_weight,
        metadata=metadata,
        head_key=head_key,
        head_source=head_location.source_key,
        vocab_mode=vocab_mode,
        vocab_start=0 if vocab_mode == "full" else int(vocab_start),
        vocab_size=int(head_weight.shape[0]),
        full_vocab_size=int(config.vocab_size),
    )


def build_torch_decode_logits_reference(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    head_weight: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
) -> dict[str, torch.Tensor]:
    _validate_hidden_states(hidden_states, hidden_size=config.hidden_size)
    _validate_logits_weights(norm_weight, head_weight, hidden_size=config.hidden_size)
    final_norm = rms_norm(hidden_states[:, 0], norm_weight.to(torch.bfloat16), eps=config.rms_norm_eps)
    final_norm = final_norm.unsqueeze(1).to(torch.bfloat16)
    logits = F.linear(final_norm.float(), head_weight.float()).to(torch.bfloat16)
    return {
        "final_norm": final_norm.contiguous(),
        "logits": logits.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real DeepSeek V4 Flash one-token decode hidden state through final norm and LM head."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_DECODE_LOGITS_LAYER)
    parser.add_argument(
        "--expert",
        type=int,
        default=None,
        help="Routed expert to materialize. Omit to choose the active expert for the decode token.",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--vocab-mode", choices=("full", "slice"), default="full")
    parser.add_argument("--vocab-start", type=int, default=0)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_DECODE_LOGITS_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_DECODE_LOGITS_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--decoder-final-pcc", type=float, default=0.99)
    parser.add_argument("--final-norm-pcc", type=float, default=0.999)
    parser.add_argument("--logits-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--decoder-residual-atol", type=float, default=3e-1)
    parser.add_argument("--logits-rtol", type=float, default=1e-1)
    parser.add_argument("--logits-atol", type=float, default=1.0)
    parser.add_argument("--top-logit-atol", type=float, default=1.0)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_decode_logits_smoke(
        args.snapshot_dir,
        layer=args.layer,
        expert=args.expert,
        prefill_seq_len=args.prefill_seq_len,
        vocab_mode=args.vocab_mode,
        vocab_start=args.vocab_start,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        decoder_final_pcc=args.decoder_final_pcc,
        final_norm_pcc=args.final_norm_pcc,
        logits_pcc=args.logits_pcc,
        rtol=args.rtol,
        atol=args.atol,
        decoder_residual_atol=args.decoder_residual_atol,
        logits_rtol=args.logits_rtol,
        logits_atol=args.logits_atol,
        top_logit_atol=args.top_logit_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_decode_logits_from_hidden(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    head_weight: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    device_id: int,
) -> dict[str, torch.Tensor]:
    import ttnn

    _validate_hidden_states(hidden_states, hidden_size=config.hidden_size)
    _validate_logits_weights(norm_weight, head_weight, hidden_size=config.hidden_size)
    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_hidden = ttnn.from_torch(
            hidden_states.contiguous(),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_norm_weight = ttnn.from_torch(
            norm_weight.contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_norm = ttnn.rms_norm(
            tt_hidden,
            weight=tt_norm_weight,
            epsilon=config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_head_weight = ttnn.from_torch(
            head_weight.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_logits = ttnn.linear(tt_norm, tt_head_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return {
            "final_norm": ttnn.to_torch(tt_norm).contiguous(),
            "logits": ttnn.to_torch(tt_logits).contiguous(),
        }
    finally:
        ttnn.close_device(device)


def _augment_base_result_for_logits(
    result: dict[str, Any],
    *,
    config: DeepSeekV4FlashConfig,
    logits_weights: DecodeLogitsWeights,
    reference: Mapping[str, torch.Tensor],
    top_k: int,
) -> None:
    result["schema_version"] = REAL_DECODE_LOGITS_SMOKE_SCHEMA_VERSION
    result["logits_scope"] = {
        "path": "one-layer decode post-FFN residual -> final RMSNorm -> LM head projection -> logits",
        "decoder_layer_source": "real one-token decode decoder-layer smoke path",
        "embeddings_vllm_evals": "excluded",
    }
    result["decoder_scope"][
        "embeddings_logits_vllm_evals"
    ] = "embeddings/vLLM/evals excluded; final norm and LM head included"
    result["model"]["vocab_size"] = int(config.vocab_size)
    result["output_shapes"]["final_norm"] = list(reference["final_norm"].shape)
    result["output_shapes"]["logits"] = list(reference["logits"].shape)
    result["vocab"] = {
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
    }
    result["final_norm_lm_head"] = {
        "loaded_keys": {
            "final_norm": "norm.weight",
            "lm_head": logits_weights.head_key,
        },
        "loaded_source_keys": {
            "final_norm": "norm.weight",
            "lm_head": logits_weights.head_source,
        },
        "payload_bytes": _head_payload_byte_split(logits_weights.metadata),
        "lm_head_shape_loaded": list(logits_weights.head_weight.shape),
        "lm_head_full_shape": [logits_weights.full_vocab_size, int(config.hidden_size)],
    }
    result["payload_bytes"]["final_norm_lm_head"] = result["final_norm_lm_head"]["payload_bytes"]
    result["host_boundaries"].extend(
        [
            {
                "name": "decoder_layer_output_readback",
                "location": "before final RMSNorm",
                "description": "the proven decoder-layer smoke exposes post-FFN residual as a host tensor",
            },
            {
                "name": "logits_readback",
                "location": "after LM head",
                "description": "logits are copied back to host for top-k and dense accuracy comparison",
            },
        ]
    )
    if logits_weights.vocab_mode == "slice":
        result["host_boundaries"].append(
            {
                "name": "lm_head_vocab_slice",
                "location": "checkpoint load",
                "description": "a deterministic row slice of the LM head is loaded from safetensors on host",
            }
        )
    result["reference_ops"].extend(
        [
            "torch.rms_norm_reference(final_norm)",
            f"torch.linear(lm_head_{logits_weights.vocab_mode})",
            "torch.topk(logits)",
        ]
    )
    result["reference"]["final_norm"] = _tensor_summary(reference["final_norm"])
    result["reference"]["logits"] = _tensor_summary(reference["logits"])
    result["reference"]["top_k"] = _topk_summary(
        reference["logits"],
        top_k=top_k,
        vocab_start=logits_weights.vocab_start,
    )


def _resolve_head_key(index: RealCheckpointTensorIndex) -> str:
    if index.has_tensor("head.weight"):
        return "head.weight"
    if index.has_tensor("embed.weight"):
        return "embed.weight"
    raise KeyError(f"Snapshot {index.snapshot_dir} has neither 'head.weight' nor tied 'embed.weight'")


def _slice_head_metadata(
    item: TensorMetadata,
    *,
    config: DeepSeekV4FlashConfig,
    vocab_mode: VocabMode,
    vocab_start: int,
    vocab_size: int | None,
) -> TensorMetadata:
    if item.canonical_key == "norm.weight" or vocab_mode == "full":
        return item
    selected_vocab_size = _resolve_vocab_size(config, vocab_start=vocab_start, vocab_size=vocab_size)
    return TensorMetadata(
        canonical_key=item.canonical_key,
        source_key=item.source_key,
        shard_name=item.shard_name,
        shard_path=item.shard_path,
        dtype=item.dtype,
        shape=(selected_vocab_size, int(config.hidden_size)),
        nbytes=_safetensors_nbytes(item.dtype, (selected_vocab_size, int(config.hidden_size))),
    )


def _metadata_for_key(metadata: Sequence[TensorMetadata], key: str) -> TensorMetadata:
    for item in metadata:
        if item.canonical_key == key:
            return item
    raise KeyError(f"Missing metadata for {key}")


def _resolve_vocab_size(config: DeepSeekV4FlashConfig, *, vocab_start: int, vocab_size: int | None) -> int:
    selected = DEFAULT_DECODE_LOGITS_VOCAB_SLICE_SIZE if vocab_size is None else int(vocab_size)
    if selected <= 0:
        raise ValueError(f"vocab_size must be positive, got {selected}")
    if vocab_start + selected > int(config.vocab_size):
        raise ValueError(
            f"vocab slice [{vocab_start}, {vocab_start + selected}) exceeds vocab size {config.vocab_size}"
        )
    return selected


def _validate_logits_args(
    *,
    vocab_mode: VocabMode,
    vocab_start: int,
    vocab_size: int | None,
    top_k: int,
) -> None:
    if vocab_mode not in ("full", "slice"):
        raise ValueError(f"vocab_mode must be 'full' or 'slice', got {vocab_mode!r}")
    if vocab_start < 0:
        raise ValueError(f"vocab_start must be non-negative, got {vocab_start}")
    if vocab_mode == "full" and vocab_start != 0:
        raise ValueError("vocab_start must be 0 in full-vocab mode")
    if vocab_mode == "full" and vocab_size is not None:
        raise ValueError("vocab_size may only be provided in slice mode")
    if vocab_size is not None and vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive when provided, got {vocab_size}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")


def _validate_logits_metadata(
    norm_metadata: TensorMetadata,
    head_metadata: TensorMetadata,
    *,
    config: DeepSeekV4FlashConfig,
    vocab_mode: VocabMode,
) -> None:
    if tuple(norm_metadata.shape) != (int(config.hidden_size),):
        raise ValueError(f"Expected norm.weight shape {(config.hidden_size,)}, got {norm_metadata.shape}")
    expected_vocab = int(config.vocab_size) if vocab_mode == "full" else int(head_metadata.shape[0])
    if tuple(head_metadata.shape) != (expected_vocab, int(config.hidden_size)):
        raise ValueError(
            f"Expected LM head shape {(expected_vocab, int(config.hidden_size))}, got {head_metadata.shape}"
        )
    allowed_dtypes = {"BF16", "F16", "F32"}
    if norm_metadata.dtype not in allowed_dtypes:
        raise ValueError(f"Unsupported final norm dtype {norm_metadata.dtype}")
    if head_metadata.dtype not in allowed_dtypes:
        raise ValueError(f"Unsupported LM head dtype {head_metadata.dtype}")


def _validate_hidden_states(hidden_states: torch.Tensor, *, hidden_size: int) -> None:
    if hidden_states.ndim != 4 or tuple(hidden_states.shape[:3]) != (1, 1, 1):
        raise ValueError(f"hidden_states must have shape [1, 1, 1, hidden], got {tuple(hidden_states.shape)}")
    if int(hidden_states.shape[-1]) != int(hidden_size):
        raise ValueError(f"hidden_states hidden size must be {hidden_size}, got {hidden_states.shape[-1]}")


def _validate_logits_weights(norm_weight: torch.Tensor, head_weight: torch.Tensor, *, hidden_size: int) -> None:
    if tuple(norm_weight.shape) != (int(hidden_size),):
        raise ValueError(f"norm_weight must have shape {(hidden_size,)}, got {tuple(norm_weight.shape)}")
    if head_weight.ndim != 2 or int(head_weight.shape[-1]) != int(hidden_size):
        raise ValueError(f"head_weight must have shape [vocab, {hidden_size}], got {tuple(head_weight.shape)}")
    if int(head_weight.shape[0]) <= 0:
        raise ValueError("head_weight must contain at least one vocab row")


def _enforce_total_budget(metadata: Sequence[TensorMetadata], *, max_tensors: int, max_bytes: int) -> None:
    if len(metadata) > max_tensors:
        raise ValueError(f"Requested {len(metadata)} tensors exceeds tensor budget {max_tensors}")
    selected_bytes = sum(item.nbytes for item in metadata)
    if selected_bytes > max_bytes:
        raise ValueError(f"Requested {selected_bytes} bytes exceeds byte budget {max_bytes}")


def _head_payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {"final_norm": 0, "lm_head": 0}
    for item in metadata:
        if item.canonical_key == "norm.weight":
            split["final_norm"] += item.nbytes
        elif item.canonical_key in ("head.weight", "embed.weight"):
            split["lm_head"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in final norm/LM head slice: {item.canonical_key}")
    split["total"] = sum(split.values())
    return split


def _dense_pcc_summary(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    pcc_threshold: float,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(expected.shape):
        return {
            "passed": False,
            "reason": f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}",
        }
    expected_float = expected.float()
    actual_float = actual.float()
    abs_diff = (actual_float - expected_float).abs()
    pcc = _pcc(expected_float, actual_float)
    allclose = bool(torch.allclose(actual_float, expected_float, rtol=rtol, atol=atol))
    return {
        "passed": bool(pcc >= pcc_threshold),
        "pcc": float(pcc),
        "pcc_threshold": float(pcc_threshold),
        "allclose": allclose,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
    }


def _topk_summary(logits: torch.Tensor, *, top_k: int, vocab_start: int) -> list[dict[str, float | int]]:
    flat = logits.reshape(-1).float()
    k = min(int(top_k), int(flat.numel()))
    values, local_ids = torch.topk(flat, k=k)
    return [
        {
            "id": int(local_id.item()) + int(vocab_start),
            "local_id": int(local_id.item()),
            "value": float(value.item()),
        }
        for value, local_id in zip(values, local_ids)
    ]


def _topk_accuracy_summary(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    top_k: int,
    vocab_start: int,
    top_logit_atol: float,
) -> dict[str, Any]:
    expected_topk = _topk_summary(expected, top_k=top_k, vocab_start=vocab_start)
    actual_topk = _topk_summary(actual, top_k=top_k, vocab_start=vocab_start)
    expected_ids = [item["id"] for item in expected_topk]
    actual_ids = [item["id"] for item in actual_topk]
    exact_order_match = expected_ids == actual_ids
    exact_set_match = set(expected_ids) == set(actual_ids)
    expected_top_logit = expected_topk[0]["value"]
    actual_top_logit = actual.reshape(-1).float()[int(expected_topk[0]["local_id"])].item()
    top_logit_abs = abs(float(actual_top_logit) - float(expected_top_logit))
    return {
        "passed": bool(expected_ids[:1] == actual_ids[:1] and exact_set_match and top_logit_abs <= top_logit_atol),
        "expected": expected_topk,
        "actual": actual_topk,
        "exact_id_match": exact_order_match,
        "exact_order_match": exact_order_match,
        "exact_set_match": exact_set_match,
        "top1_match": expected_ids[:1] == actual_ids[:1],
        "top_logit_atol": float(top_logit_atol),
        "expected_top_logit_actual_value": float(actual_top_logit),
        "top_logit_max_abs": float(top_logit_abs),
    }


if __name__ == "__main__":
    main()
