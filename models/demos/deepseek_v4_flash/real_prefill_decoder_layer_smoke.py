# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import deterministic_attention_activation
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_EXPERT_MLP_LAYER,
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_router_norm_keys,
)
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights
from models.demos.deepseek_v4_flash.real_ffn_fanout_smoke import (
    DEFAULT_FFN_FANOUT_MAX_BYTES,
    _fanout_summary,
    _ordered_activated_expert_ids,
)
from models.demos.deepseek_v4_flash.real_ffn_fanout_smoke import _payload_byte_split as _ffn_fanout_payload_byte_split
from models.demos.deepseek_v4_flash.real_ffn_fanout_smoke import (
    _run_ttnn_ffn_fanout_slice,
    build_torch_ffn_fanout_reference,
    build_torch_ffn_fanout_selector_reference,
    layer_ffn_fanout_keys,
    validate_real_ffn_fanout_slice,
)
from models.demos.deepseek_v4_flash.real_ffn_smoke import (
    DEFAULT_FFN_MAX_TENSORS,
    _accuracy_summary,
    _index_accuracy_summary,
    _metadata_summary,
    _selection_summary,
    _tensor_summary,
)
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import decode_real_kv_projection_weights
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES,
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS,
    PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE,
    _floating_accuracy_summary,
    _int_equality_summary,
    _int_tensor_summary,
)
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    _payload_byte_split as _attention_payload_byte_split,
)
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    _query_projection_weights_only,
    _run_ttnn_attention_runtime_from_cache_boundary,
    _topk_activity_summary,
    build_torch_prefill_attention_runtime_reference,
    decode_real_prefill_attention_projection_weights,
    decode_real_prefill_runtime_sparse_weights_if_needed,
    layer_prefill_attention_runtime_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    DEFAULT_SEQUENCE_LENGTH,
    _run_ttnn_prefill_cache_prep,
)
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import (
    ROUTED_MOE_TTNN_TILE_MULTIPLE,
    deterministic_input_ids_for_expert,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_real_shared_expert_weights

REAL_PREFILL_DECODER_LAYER_SMOKE_SCHEMA_VERSION = 1
DEFAULT_PREFILL_DECODER_LAYER = DEFAULT_LAYER_EXPERT_MLP_LAYER
DEFAULT_PREFILL_DECODER_LAYER_EXPERT: int | None = None
DEFAULT_PREFILL_FFN_FANOUT_MAX_TENSORS = max(DEFAULT_FFN_MAX_TENSORS, 768)
DEFAULT_PREFILL_FFN_FANOUT_MAX_BYTES = max(DEFAULT_FFN_FANOUT_MAX_BYTES, 2 * 1024 * 1024 * 1024)
DEFAULT_PREFILL_DECODER_LAYER_MAX_TENSORS = (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS + DEFAULT_PREFILL_FFN_FANOUT_MAX_TENSORS + 8
)
DEFAULT_PREFILL_DECODER_LAYER_MAX_BYTES = (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES + DEFAULT_PREFILL_FFN_FANOUT_MAX_BYTES
)


def run_real_prefill_decoder_layer_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_PREFILL_DECODER_LAYER,
    expert: int | None = DEFAULT_PREFILL_DECODER_LAYER_EXPERT,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    start_pos: int = 0,
    max_tensors: int = DEFAULT_PREFILL_DECODER_LAYER_MAX_TENSORS,
    max_bytes: int = DEFAULT_PREFILL_DECODER_LAYER_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    attention_pcc: float = 0.99,
    output_pcc: float = 0.99,
    residual_pcc: float = 0.99,
    ffn_norm_pcc: float = 0.999,
    router_pcc: float = 0.99,
    router_index_match: float = 0.7,
    routed_pcc: float = 0.99,
    shared_pcc: float = 0.99,
    combined_pcc: float = 0.99,
    final_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    attention_atol: float = 2e-1,
    residual_atol: float = 3e-1,
) -> dict[str, Any]:
    """Run one real DeepSeek V4 Flash prefill decoder-layer slice.

    The slice wires the already-proven real prefill attention runtime into the
    real full-routed-fanout FFN composition:
    input -> attn_norm/attention -> residual -> ffn_norm/router/all activated routed experts/shared expert -> residual.
    """

    _validate_smoke_args(
        layer=layer,
        expert=expert,
        seq_len=seq_len,
        start_pos=start_pos,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        pcc_thresholds={
            "attention_pcc": attention_pcc,
            "output_pcc": output_pcc,
            "residual_pcc": residual_pcc,
            "ffn_norm_pcc": ffn_norm_pcc,
            "router_pcc": router_pcc,
            "router_index_match": router_index_match,
            "routed_pcc": routed_pcc,
            "shared_pcc": shared_pcc,
            "combined_pcc": combined_pcc,
            "final_pcc": final_pcc,
        },
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    _validate_runtime_config(config, layer=layer, seq_len=seq_len, start_pos=start_pos)
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
    compressor_weights, indexer_weights = decode_real_prefill_runtime_sparse_weights_if_needed(
        tensors,
        config=config,
        layer=layer,
        seq_len=seq_len,
    )

    activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    attention_reference = build_torch_prefill_attention_runtime_reference(
        tensors,
        q_weights,
        kv_weights,
        compressor_weights,
        indexer_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=start_pos,
    )
    post_attention_residual = _residual_add(activation, attention_reference["attention_output_projected"])
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
    ) = _prepare_prefill_ffn_fanout(
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
    reference = build_torch_prefill_decoder_layer_reference(
        activation=activation,
        attention_reference=attention_reference,
        ffn_reference=ffn_reference,
    )
    metadata_groups = _metadata_groups(
        metadata,
        attention_keys=attention_keys,
        selector_keys=selector_keys,
        ffn_keys=ffn_keys,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        activated_experts=activated_experts,
        requested_expert=expert,
        ffn_input_ids=ffn_input_ids,
        seq_len=seq_len,
        start_pos=start_pos,
        config=config,
        metadata=metadata,
        metadata_groups=metadata_groups,
        activation=activation,
        reference=reference,
        router_preview=router_preview,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
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

    if seq_len % PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(
            f"TTNN smoke seq_len must be a multiple of {PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE}, got {seq_len}"
        )
    if seq_len % ROUTED_MOE_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN FFN seq_len must be a multiple of {ROUTED_MOE_TTNN_TILE_MULTIPLE}, got {seq_len}")

    ttnn_outputs = _run_ttnn_prefill_decoder_layer_slice(
        tensors,
        q_weights,
        kv_weights,
        compressor_weights,
        indexer_weights,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=layer,
        input_ids=ffn_input_ids,
        activation=activation,
        start_pos=start_pos,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "ttnn.rms_norm(attn_norm)",
        "TtAttentionProjection.project_q_rank",
        "ttnn.linear(wq_a)",
        "ttnn.rms_norm(q_norm)",
        "TtAttentionProjection.project_q_from_rank",
        "ttnn.linear(wq_b)",
        "ttnn.linear(wkv)",
        "ttnn.rms_norm(kv_norm)",
        "ttnn.reshape/slice(q and kv heads)",
        "TtPrefillCompressor.build_compressed_kv_cache",
        "TtSparsePrefillAttention",
        "ttnn.linear(wo_b)",
        "host_add(input_hidden_states,attention_output_projected)",
        "ttnn.rms_norm(ffn_norm)",
        "TtRouter(ttnn.linear+host_topk)",
        "host_gather_activated_expert_tokens",
        "sequential_TtRoutedExpertMLP_per_activated_expert",
        "host_scatter_add_activated_expert_outputs",
        "TtSharedExpertMLP",
        "host_add(full_routed_output,shared_expert_output)",
        "host_add(post_attention_residual,combined_ffn_output)",
    ]
    result["ttnn"] = {
        "attention_output": _tensor_summary(ttnn_outputs["attention_output"]),
        "attention_output_projected": _tensor_summary(ttnn_outputs["attention_output_projected"]),
        "post_attention_residual": _tensor_summary(ttnn_outputs["post_attention_residual"]),
        "ffn_norm": _tensor_summary(ttnn_outputs["ffn_norm_output"]),
        "router_weights": _tensor_summary(ttnn_outputs["router_weights"]),
        "router_indices": _tensor_summary(ttnn_outputs["router_indices"]),
        "activated_experts": _fanout_summary(
            ttnn_outputs["router_weights"],
            ttnn_outputs["router_indices"],
            full_topk=config.num_experts_per_tok,
        ),
        "per_expert_routes": {
            str(expert_id): _selection_summary(route) for expert_id, route in ttnn_outputs["per_expert_routes"].items()
        },
        "per_expert_padding": {
            str(expert_id): padding for expert_id, padding in ttnn_outputs["per_expert_padding"].items()
        },
        "per_expert_selected_output": {
            str(expert_id): _tensor_summary(output)
            for expert_id, output in ttnn_outputs["per_expert_selected_output"].items()
        },
        "routed_output": _tensor_summary(ttnn_outputs["routed_output"]),
        "shared_output": _tensor_summary(ttnn_outputs["shared_output"]),
        "combined_ffn_output": _tensor_summary(ttnn_outputs["combined_ffn_output"]),
        "post_ffn_residual": _tensor_summary(ttnn_outputs["post_ffn_residual"]),
        "experts_executed": int(len(ttnn_outputs["per_expert_routes"])),
        "routes_executed": int(ttnn_outputs["router_indices"].numel()),
        "input_padding": ttnn_outputs["input_padding"],
    }
    result["ttnn_int"] = {
        "window_topk_idxs": _int_tensor_summary(ttnn_outputs["window_topk_idxs"]),
        "compress_topk_idxs": _int_tensor_summary(ttnn_outputs["compress_topk_idxs"]),
        "runtime_topk_idxs": _int_tensor_summary(ttnn_outputs["runtime_topk_idxs"]),
    }
    accuracy = {
        "attention_output": _floating_accuracy_summary(
            reference["attention"]["attention_output"],
            ttnn_outputs["attention_output"],
            pcc_threshold=attention_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "attention_output_projected": _accuracy_summary(
            reference["attention"]["attention_output_projected"],
            ttnn_outputs["attention_output_projected"],
            pcc_threshold=output_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "post_attention_residual": _accuracy_summary(
            reference["post_attention_residual"],
            ttnn_outputs["post_attention_residual"],
            pcc_threshold=residual_pcc,
            rtol=rtol,
            atol=residual_atol,
        ),
        "ffn_norm": _accuracy_summary(
            reference["ffn"]["norm_output"],
            ttnn_outputs["ffn_norm_output"],
            pcc_threshold=ffn_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "router_weights": _accuracy_summary(
            reference["ffn"]["router_weights"],
            ttnn_outputs["router_weights"],
            pcc_threshold=router_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "router_indices": _index_accuracy_summary(
            reference["ffn"]["router_indices"],
            ttnn_outputs["router_indices"],
            match_threshold=router_index_match,
        ),
        "routed_output": _accuracy_summary(
            reference["ffn"]["routed_output"],
            ttnn_outputs["routed_output"],
            pcc_threshold=routed_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "shared_output": _accuracy_summary(
            reference["ffn"]["shared_output"],
            ttnn_outputs["shared_output"],
            pcc_threshold=shared_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "combined_ffn_output": _accuracy_summary(
            reference["ffn"]["combined_output"],
            ttnn_outputs["combined_ffn_output"],
            pcc_threshold=combined_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "post_ffn_residual": _accuracy_summary(
            reference["post_ffn_residual"],
            ttnn_outputs["post_ffn_residual"],
            pcc_threshold=final_pcc,
            rtol=rtol,
            atol=residual_atol,
        ),
        "window_topk_idxs": _int_equality_summary(
            reference["attention"]["window_topk_idxs"], ttnn_outputs["window_topk_idxs"]
        ),
    }
    for name in ("compress_topk_idxs", "runtime_topk_idxs"):
        if indexer_weights is None:
            accuracy[name] = _int_equality_summary(reference["attention"][name], ttnn_outputs[name])
        else:
            accuracy[name] = _topk_activity_summary(reference["attention"][name], ttnn_outputs[name])
    result["accuracy"] = accuracy
    result["router_match_stats"] = {
        "index_match_fraction": result["accuracy"]["router_indices"].get("match_fraction"),
        "index_mismatch_count": result["accuracy"]["router_indices"].get("mismatch_count"),
        "weights_pcc": result["accuracy"]["router_weights"].get("pcc"),
        "weights_max_abs": result["accuracy"]["router_weights"].get("max_abs"),
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_prefill_decoder_layer_keys(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
) -> list[str]:
    return _unique_keys(
        [
            *layer_prefill_attention_runtime_keys(index, config=config, layer=layer),
            *layer_ffn_fanout_keys(index, layer=layer, experts=(expert,)),
        ]
    )


def layer_prefill_decoder_layer_selector_keys(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> list[str]:
    return _unique_keys(
        [
            *layer_prefill_attention_runtime_keys(index, config=config, layer=layer),
            *layer_router_norm_keys(index, layer=layer),
        ]
    )


def build_torch_prefill_decoder_layer_reference(
    *,
    activation: torch.Tensor,
    attention_reference: Mapping[str, torch.Tensor],
    ffn_reference: Mapping[str, Any],
) -> dict[str, Any]:
    post_attention_residual = _residual_add(activation, attention_reference["attention_output_projected"])
    if tuple(ffn_reference["residual_output"].shape) != tuple(post_attention_residual.shape):
        raise ValueError(
            "FFN fanout residual shape mismatch: "
            f"{tuple(ffn_reference['residual_output'].shape)} vs {tuple(post_attention_residual.shape)}"
        )
    return {
        "input_hidden_states": activation,
        "attention": attention_reference,
        "post_attention_residual": post_attention_residual,
        "ffn": ffn_reference,
        "post_ffn_residual": ffn_reference["residual_output"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real DeepSeek V4 Flash prefill decoder-layer TTNN composition slice."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_PREFILL_DECODER_LAYER)
    parser.add_argument(
        "--expert",
        type=int,
        default=None,
        help="Expert used to choose deterministic input ids for hash-routed layers; ignored for bias-routed layers.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_PREFILL_DECODER_LAYER_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_PREFILL_DECODER_LAYER_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--attention-pcc", type=float, default=0.99)
    parser.add_argument("--output-pcc", type=float, default=0.99)
    parser.add_argument("--residual-pcc", type=float, default=0.99)
    parser.add_argument("--ffn-norm-pcc", type=float, default=0.999)
    parser.add_argument("--router-pcc", type=float, default=0.99)
    parser.add_argument("--router-index-match", type=float, default=0.7)
    parser.add_argument("--routed-pcc", type=float, default=0.99)
    parser.add_argument("--shared-pcc", type=float, default=0.99)
    parser.add_argument("--combined-pcc", type=float, default=0.99)
    parser.add_argument("--final-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--attention-atol", type=float, default=2e-1)
    parser.add_argument("--residual-atol", type=float, default=3e-1)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_prefill_decoder_layer_smoke(
        args.snapshot_dir,
        layer=args.layer,
        expert=args.expert,
        seq_len=args.seq_len,
        start_pos=args.start_pos,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        attention_pcc=args.attention_pcc,
        output_pcc=args.output_pcc,
        residual_pcc=args.residual_pcc,
        ffn_norm_pcc=args.ffn_norm_pcc,
        router_pcc=args.router_pcc,
        router_index_match=args.router_index_match,
        routed_pcc=args.routed_pcc,
        shared_pcc=args.shared_pcc,
        combined_pcc=args.combined_pcc,
        final_pcc=args.final_pcc,
        rtol=args.rtol,
        atol=args.atol,
        attention_atol=args.attention_atol,
        residual_atol=args.residual_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _load_attention_and_ffn_selector_slice(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    max_tensors: int,
    max_bytes: int,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata], list[str], list[str]]:
    attention_keys = layer_prefill_attention_runtime_keys(index, config=config, layer=layer)
    selector_keys = layer_router_norm_keys(index, layer=layer)
    keys = _unique_keys([*attention_keys, *selector_keys])
    tensors, metadata = index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)
    return tensors, metadata, attention_keys, selector_keys


def _load_missing_tensors(
    index: RealCheckpointTensorIndex,
    tensors: dict[str, torch.Tensor],
    metadata: list[TensorMetadata],
    keys: Sequence[str],
    *,
    max_tensors: int,
    max_bytes: int,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    missing_keys = [key for key in keys if key not in tensors]
    if not missing_keys:
        return tensors, metadata
    used_bytes = sum(item.nbytes for item in metadata)
    remaining_tensors = max_tensors - len(metadata)
    remaining_bytes = max_bytes - used_bytes
    loaded_tensors, loaded_metadata = index.load_tensors(
        missing_keys,
        max_tensors=remaining_tensors,
        max_bytes=remaining_bytes,
    )
    tensors.update(loaded_tensors)
    return tensors, [*metadata, *loaded_metadata]


def _prepare_prefill_ffn_fanout(
    index: RealCheckpointTensorIndex,
    tensors: dict[str, torch.Tensor],
    metadata: list[TensorMetadata],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    requested_expert: int | None,
    activation: torch.Tensor,
    max_tensors: int,
    max_bytes: int,
) -> tuple[
    dict[str, torch.Tensor],
    list[TensorMetadata],
    list[str],
    list[int],
    dict[str, Any],
    dict[int, Mapping[str, torch.Tensor]],
    Mapping[str, torch.Tensor],
    dict[str, Any],
    torch.Tensor | None,
]:
    input_ids = _prefill_fanout_input_ids_for_activation(
        tensors,
        layer=layer,
        requested_expert=requested_expert,
        seq_len=int(activation.shape[-2]),
    )
    selector_reference = build_torch_ffn_fanout_selector_reference(
        tensors,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
    )
    activated_experts = _ordered_activated_expert_ids(selector_reference["router_indices"])
    ffn_keys = layer_ffn_fanout_keys(index, layer=layer, experts=activated_experts)
    tensors, metadata = _load_missing_tensors(
        index,
        tensors,
        metadata,
        ffn_keys,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    validate_real_ffn_fanout_slice(tensors, config=config, layer=layer, experts=activated_experts)
    routed_weights_by_expert = {
        expert_id: decode_real_expert_weights(tensors, config=config, layer=layer, expert=expert_id)
        for expert_id in activated_experts
    }
    shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=layer)
    ffn_reference = build_torch_ffn_fanout_reference(
        tensors,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
    )
    router_preview = _prefill_fanout_preview(
        requested_expert=requested_expert,
        input_ids=input_ids,
        reference=ffn_reference,
        config=config,
    )
    return (
        tensors,
        metadata,
        ffn_keys,
        activated_experts,
        router_preview,
        routed_weights_by_expert,
        shared_weights,
        ffn_reference,
        input_ids,
    )


def _prefill_fanout_input_ids_for_activation(
    tensors: Mapping[str, torch.Tensor],
    *,
    layer: int,
    requested_expert: int | None,
    seq_len: int,
) -> torch.Tensor | None:
    tid2eid = tensors.get(f"layers.{layer}.ffn.gate.tid2eid")
    if tid2eid is None:
        return None
    if requested_expert is None:
        raise ValueError("Hash-routed prefill FFN fanout requires --expert to choose deterministic input ids")
    return deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=int(requested_expert), seq_len=seq_len)


def _prefill_fanout_preview(
    *,
    requested_expert: int | None,
    input_ids: torch.Tensor | None,
    reference: Mapping[str, Any],
    config: DeepSeekV4FlashConfig,
) -> dict[str, Any]:
    router_weights = reference["router_weights"]
    router_indices = reference["router_indices"]
    counts = _expert_route_counts(router_indices, n_routed_experts=config.n_routed_experts)
    return {
        "source": "torch_router_full_topk_on_post_attention_prefill_residual",
        "requested_expert": requested_expert,
        "input_id_anchor_expert": requested_expert if input_ids is not None else None,
        "topk": int(config.num_experts_per_tok),
        "routes_executed": int(router_indices.numel()),
        "activated_experts": _fanout_summary(router_weights, router_indices, full_topk=config.num_experts_per_tok),
        "top_expert_counts": counts[:16],
        "per_expert_routes": {
            str(expert_id): _selection_summary(values["route"]) for expert_id, values in reference["per_expert"].items()
        },
    }


def _run_ttnn_prefill_decoder_layer_slice(
    tensors: Mapping[str, torch.Tensor],
    q_weights,
    kv_weights,
    compressor_weights,
    indexer_weights,
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    input_ids: torch.Tensor | None,
    activation: torch.Tensor,
    start_pos: int,
    device_id: int,
) -> dict[str, Any]:
    cache_q_weights = _query_projection_weights_only(q_weights)
    cache_outputs = _run_ttnn_prefill_cache_prep(
        tensors,
        cache_q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=start_pos,
        device_id=device_id,
    )
    attention_outputs = _run_ttnn_attention_runtime_from_cache_boundary(
        cache_outputs,
        q_weights,
        compressor_weights,
        indexer_weights,
        tensors[f"layers.{layer}.attn.attn_sink"],
        config=config,
        layer=layer,
        start_pos=start_pos,
        device_id=device_id,
    )
    post_attention_residual = _residual_add(activation, attention_outputs["attention_output_projected"])
    ffn_outputs = _run_ttnn_ffn_fanout_slice(
        tensors,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=layer,
        activation=post_attention_residual,
        input_ids=input_ids,
        device_id=device_id,
    )
    return {
        **cache_outputs,
        **attention_outputs,
        "post_attention_residual": post_attention_residual,
        "ffn_norm_output": ffn_outputs["norm_output"],
        "router_weights": ffn_outputs["router_weights"],
        "router_indices": ffn_outputs["router_indices"],
        "per_expert_routes": ffn_outputs["per_expert_routes"],
        "per_expert_padding": ffn_outputs["per_expert_padding"],
        "per_expert_selected_output": ffn_outputs["per_expert_selected_output"],
        "routed_output": ffn_outputs["routed_output"],
        "shared_output": ffn_outputs["shared_output"],
        "combined_ffn_output": ffn_outputs["combined_output"],
        "post_ffn_residual": ffn_outputs["residual_output"],
        "input_padding": ffn_outputs["input_padding"],
    }


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    activated_experts: Sequence[int],
    requested_expert: int | None,
    ffn_input_ids: torch.Tensor | None,
    seq_len: int,
    start_pos: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    metadata_groups: Mapping[str, Sequence[TensorMetadata]],
    activation: torch.Tensor,
    reference: dict[str, Any],
    router_preview: dict[str, Any],
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    attention_metadata = metadata_groups["attention_runtime"]
    ffn_metadata = metadata_groups["ffn"]
    attention_payload = _attention_payload_byte_split(attention_metadata)
    ffn_payload = _ffn_fanout_payload_byte_split(ffn_metadata)
    total_payload = sum(item.nbytes for item in metadata)
    return {
        "schema_version": REAL_PREFILL_DECODER_LAYER_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "requested_expert": requested_expert,
        "input_id_anchor_expert": requested_expert if ffn_input_ids is not None else None,
        "sequence_length": int(seq_len),
        "start_pos": int(start_pos),
        "model": {
            "hidden_size": config.hidden_size,
            "q_lora_rank": config.q_lora_rank,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "o_groups": config.o_groups,
            "o_lora_rank": config.o_lora_rank,
            "compress_ratio": int(config.compress_ratios[layer]),
            "sliding_window": config.sliding_window,
            "moe_intermediate_size": config.moe_intermediate_size,
            "n_routed_experts": config.n_routed_experts,
            "n_shared_experts": config.n_shared_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "scoring_func": config.scoring_func,
            "routed_scaling_factor": config.routed_scaling_factor,
            "rms_norm_eps": config.rms_norm_eps,
            "swiglu_limit": config.swiglu_limit,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "decoder_scope": {
            "path": (
                "input hidden states -> attn_norm + real prefill attention runtime -> attention residual -> "
                "ffn_norm + router + all activated routed experts + shared expert -> FFN residual"
            ),
            "layer_choice": "layer 3 is the default because real attention runtime and real FFN are proven there",
            "full_expert_fanout": "enabled for every router top-k expert selected by every prefill token",
            "embeddings_logits_vllm_evals": "excluded",
        },
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "loaded_tensor_groups": {
            name: {
                "count": len(items),
                "payload_bytes": sum(item.nbytes for item in items),
                "canonical_keys": [item.canonical_key for item in items],
            }
            for name, items in metadata_groups.items()
        },
        "payload_bytes": {
            "attention_runtime": attention_payload,
            "ffn": ffn_payload,
            "total": total_payload,
        },
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": total_payload,
        },
        "output_shapes": {
            "input_hidden_states": list(activation.shape),
            "attention_output": list(reference["attention"]["attention_output"].shape),
            "attention_output_projected": list(reference["attention"]["attention_output_projected"].shape),
            "post_attention_residual": list(reference["post_attention_residual"].shape),
            "ffn_norm": list(reference["ffn"]["norm_output"].shape),
            "per_expert_selected_output": {
                str(expert_id): list(values["selected_expert_output"].shape)
                for expert_id, values in reference["ffn"]["per_expert"].items()
            },
            "routed_output": list(reference["ffn"]["routed_output"].shape),
            "shared_output": list(reference["ffn"]["shared_output"].shape),
            "combined_ffn_output": list(reference["ffn"]["combined_output"].shape),
            "post_ffn_residual": list(reference["post_ffn_residual"].shape),
        },
        "prefill_fanout_info": router_preview,
        "fanout_scope": {
            "full_expert_fanout": True,
            "activated_expert_ids": [int(expert_id) for expert_id in activated_experts],
            "activated_expert_count": len(activated_experts),
            "topk": int(config.num_experts_per_tok),
            "routes_executed": int(reference["ffn"]["router_indices"].numel()),
            "tokens": int(seq_len),
            "unique_expert_cap": None,
        },
        "sparse_attention_inputs": {
            "window_topk_idxs": _int_tensor_summary(reference["attention"]["window_topk_idxs"]),
            "compress_topk_idxs": _int_tensor_summary(reference["attention"]["compress_topk_idxs"]),
            "runtime_topk_idxs": _int_tensor_summary(reference["attention"]["runtime_topk_idxs"]),
            "compressed_cache_length": int(reference["attention"]["compressed_kv"].shape[1]),
            "compressed_topk_width": int(reference["attention"]["compress_topk_idxs"].shape[-1]),
        },
        "host_boundaries": [
            {
                "name": "projection_fp8_decode_to_bf16",
                "location": "before TTNN attention projection modules",
                "description": "real FP8 Q, K/V, wo_a, and wo_b weights are decoded on host to BF16",
            },
            {
                "name": "cache_prep_readback",
                "location": "after real Q/KV prefill cache prep",
                "description": "q_prefill and kv_cache_ready are copied to host at the existing cache-prep boundary",
            },
            {
                "name": "sparse_attention_host_fallback",
                "location": "inside TtSparsePrefillAttention",
                "description": "indexed gather, attention-sink softmax, and weighted reduction currently run on host",
            },
            {
                "name": "inverse_rope_host",
                "location": "after sparse attention",
                "description": "attention output RoPE dimensions are inverse-rotated on host before output projection",
            },
            {
                "name": "grouped_wo_a_host",
                "location": "inside output projection",
                "description": "grouped wo_a projection runs on host before TTNN wo_b",
            },
            {
                "name": "attention_residual_host_add",
                "location": "decoder layer attention residual",
                "description": "attention projection output is added to the input hidden states on host",
            },
            {
                "name": "activated_expert_slice_selection",
                "location": "before loading routed expert weights",
                "description": "all unique experts selected by the post-attention router preview are loaded",
            },
            {
                "name": "routed_fp4_decode_to_bf16",
                "location": "before TtRoutedExpertMLP",
                "description": "packed-FP4 routed expert weights and scales are decoded on host to BF16",
            },
            {
                "name": "router_topk",
                "location": "TtRouter",
                "description": "router scores leave device for host DeepSeek top-k/hash selection",
            },
            {
                "name": "activated_expert_gather",
                "location": "between router and routed expert",
                "description": "token activations and route weights are gathered on host for each activated expert",
            },
            {
                "name": "activated_expert_scatter_add",
                "location": "after TtRoutedExpertMLP",
                "description": "activated expert contributions are scattered back to full sequence shape and accumulated on host",
            },
            {
                "name": "ffn_host_combine",
                "location": "after routed and shared experts",
                "description": "full routed fanout and shared expert contributions are added on host",
            },
            {
                "name": "ffn_residual_host_add",
                "location": "decoder layer FFN residual",
                "description": "combined FFN contribution is added to the post-attention residual on host",
            },
        ],
        "reference_ops": [
            "torch.rms_norm_reference(attn_norm)",
            "torch.real_prefill_attention_runtime_reference",
            "host_add(input_hidden_states,attention_output_projected)",
            "torch.rms_norm_reference(ffn_norm)",
            "torch.router_reference",
            "host_gather_activated_expert_tokens",
            "torch.routed_swiglu_expert_reference_per_activated_expert",
            "host_scatter_add_activated_expert_outputs",
            "torch.shared_swiglu_expert_reference",
            "host_add(full_routed_output,shared_expert_output)",
            "host_add(post_attention_residual,combined_ffn_output)",
        ],
        "ttnn_ops": [],
        "inputs": {
            "hidden_states": _tensor_summary(activation),
            "ffn_input_ids": _tensor_summary(ffn_input_ids),
        },
        "reference": {
            "attention_output": _tensor_summary(reference["attention"]["attention_output"]),
            "attention_output_projected": _tensor_summary(reference["attention"]["attention_output_projected"]),
            "post_attention_residual": _tensor_summary(reference["post_attention_residual"]),
            "ffn_norm": _tensor_summary(reference["ffn"]["norm_output"]),
            "router_weights": _tensor_summary(reference["ffn"]["router_weights"]),
            "router_indices": _tensor_summary(reference["ffn"]["router_indices"]),
            "activated_experts": _fanout_summary(
                reference["ffn"]["router_weights"],
                reference["ffn"]["router_indices"],
                full_topk=config.num_experts_per_tok,
            ),
            "per_expert_routes": {
                str(expert_id): _selection_summary(values["route"])
                for expert_id, values in reference["ffn"]["per_expert"].items()
            },
            "per_expert_selected_output": {
                str(expert_id): _tensor_summary(values["selected_expert_output"])
                for expert_id, values in reference["ffn"]["per_expert"].items()
            },
            "routed_output": _tensor_summary(reference["ffn"]["routed_output"]),
            "shared_output": _tensor_summary(reference["ffn"]["shared_output"]),
            "combined_ffn_output": _tensor_summary(reference["ffn"]["combined_output"]),
            "post_ffn_residual": _tensor_summary(reference["post_ffn_residual"]),
        },
        "router_match_stats": {},
        "ttnn": {},
        "ttnn_int": {},
        "accuracy": {},
        "passed": False,
    }


def _metadata_groups(
    metadata: Sequence[TensorMetadata],
    *,
    attention_keys: Sequence[str],
    selector_keys: Sequence[str],
    ffn_keys: Sequence[str],
) -> dict[str, list[TensorMetadata]]:
    attention_set = set(attention_keys)
    selector_set = set(selector_keys)
    ffn_set = set(ffn_keys)
    return {
        "attention_runtime": [item for item in metadata if item.canonical_key in attention_set],
        "ffn_selector": [
            item for item in metadata if item.canonical_key in selector_set and item.canonical_key in ffn_set
        ],
        "ffn": [item for item in metadata if item.canonical_key in ffn_set],
    }


def _expert_route_counts(router_indices: torch.Tensor, *, n_routed_experts: int) -> list[dict[str, int]]:
    flat = router_indices.reshape(-1).to(torch.long)
    counts = []
    for expert in torch.unique(flat, sorted=True):
        expert_id = int(expert.item())
        if expert_id < 0 or expert_id >= n_routed_experts:
            continue
        mask = router_indices.to(torch.long) == expert_id
        counts.append(
            {
                "expert": expert_id,
                "hit_count": int(mask.sum().item()),
                "selected_token_count": int(mask.any(dim=-1).sum().item()),
            }
        )
    return sorted(counts, key=lambda item: (-item["selected_token_count"], -item["hit_count"], item["expert"]))


def _residual_add(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if tuple(residual.shape) != tuple(update.shape):
        raise ValueError(f"Residual add shape mismatch: {tuple(residual.shape)} vs {tuple(update.shape)}")
    return (residual.float() + update.float()).to(residual.dtype)


def _unique_keys(keys: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for key in keys:
        if key not in unique:
            unique.append(key)
    return unique


def _validate_runtime_config(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    seq_len: int,
    start_pos: int,
) -> None:
    if layer >= len(config.compress_ratios):
        raise ValueError(f"layer {layer} is outside compress_ratios length {len(config.compress_ratios)}")
    if start_pos != 0:
        raise ValueError("This prefill decoder-layer smoke only supports start_pos=0")
    if seq_len > int(config.sliding_window):
        raise ValueError(
            f"This first decoder-layer smoke supports seq_len <= sliding_window {config.sliding_window}, got {seq_len}"
        )
    if int(config.num_key_value_heads) != 1:
        raise ValueError(f"Expected one K/V head, got {config.num_key_value_heads}")


def _validate_smoke_args(
    *,
    layer: int,
    expert: int | None,
    seq_len: int,
    start_pos: int,
    max_tensors: int,
    max_bytes: int,
    pcc_thresholds: Mapping[str, float],
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if expert is not None and expert < 0:
        raise ValueError(f"expert must be non-negative when provided, got {expert}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if start_pos != 0:
        raise ValueError("This prefill decoder-layer smoke only supports start_pos=0")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


if __name__ == "__main__":
    main()
