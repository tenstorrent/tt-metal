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
import torch.nn.functional as F

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import compress_topk_indices, indexer_topk, sparse_attention
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import (
    build_torch_attention_projection_reference,
    deterministic_attention_activation,
)
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_EXPERT_MLP_LAYER,
    RealCheckpointTensorIndex,
    TensorMetadata,
)
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights
from models.demos.deepseek_v4_flash.real_ffn_fanout_smoke import (
    DEFAULT_FFN_FANOUT_MAX_BYTES,
    DEFAULT_FFN_FANOUT_MAX_TENSORS,
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
    _accuracy_summary,
    _index_accuracy_summary,
    _metadata_summary,
    _selection_summary,
    _tensor_summary,
    layer_ffn_keys,
)
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import (
    KvProjectionWeights,
    build_torch_kv_projection_reference,
    decode_real_kv_projection_weights,
)
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES,
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS,
    PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE,
    PrefillIndexerWeights,
    _empty_indexer_reference,
    _floating_accuracy_summary,
    _int_equality_summary,
    _int_tensor_summary,
    _inverse_attention_rope,
    _offset_valid_topk_indices,
)
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    _payload_byte_split as _attention_payload_byte_split,
)
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    _query_projection_weights_only,
    _rotate_indexer_q,
    decode_real_prefill_attention_projection_weights,
    layer_prefill_attention_runtime_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    DEFAULT_SEQUENCE_LENGTH,
    _run_ttnn_prefill_cache_prep,
    _unit_rms_norm,
    apply_deepseek_v4_rotary,
    build_torch_prefill_cache_prep_reference,
    precompute_deepseek_v4_rope_frequencies,
)
from models.demos.deepseek_v4_flash.real_prefill_decoder_layer_smoke import (
    _load_attention_and_ffn_selector_slice,
    _load_missing_tensors,
    _metadata_groups,
    _residual_add,
    _unique_keys,
)
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import deterministic_input_ids_for_expert
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_real_shared_expert_weights
from models.demos.deepseek_v4_flash.ttnn_attention_projection import (
    AttentionProjectionWeights,
    TtAttentionProjection,
    grouped_output_projection_a,
)
from models.demos.deepseek_v4_flash.ttnn_sparse_attention import TtSparsePrefillAttention

REAL_DECODE_DECODER_LAYER_SMOKE_SCHEMA_VERSION = 1
DEFAULT_DECODE_DECODER_LAYER = DEFAULT_LAYER_EXPERT_MLP_LAYER
DEFAULT_DECODE_DECODER_LAYER_EXPERT: int | None = None
DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS = (
    DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS + DEFAULT_FFN_FANOUT_MAX_TENSORS + 8
)
DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES = DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES + DEFAULT_FFN_FANOUT_MAX_BYTES


def run_real_decode_decoder_layer_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_DECODE_DECODER_LAYER,
    expert: int | None = DEFAULT_DECODE_DECODER_LAYER_EXPERT,
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS,
    max_bytes: int = DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    cache_prep_pcc: float = 0.99,
    attention_pcc: float = 0.99,
    output_pcc: float = 0.99,
    residual_pcc: float = 0.99,
    ffn_norm_pcc: float = 0.999,
    router_pcc: float = 0.99,
    router_index_match: float = 0.5,
    routed_pcc: float = 0.99,
    shared_pcc: float = 0.99,
    combined_pcc: float = 0.99,
    final_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    attention_atol: float = 2e-1,
    residual_atol: float = 3e-1,
) -> dict[str, Any]:
    """Run one real DeepSeek V4 Flash decode decoder-layer slice.

    This first slice proves the batch-1, one-token layer composition after a
    prefill-built host-visible cache. It intentionally targets layer 3 before the
    first compressed block, so the decode attention cache is the sliding-window
    KV cache from prefill plus the current token KV.
    """

    _validate_smoke_args(
        layer=layer,
        expert=expert,
        prefill_seq_len=prefill_seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        pcc_thresholds={
            "cache_prep_pcc": cache_prep_pcc,
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
    reference = {
        "prefill_input_hidden_states": prefill_activation,
        "decode_input_hidden_states": decode_activation,
        "prefill_cache": prefill_cache_reference,
        "decode_cache": decode_cache_reference,
        "attention": attention_reference,
        "post_attention_residual": post_attention_residual,
        "ffn": ffn_reference,
        "post_ffn_residual": ffn_reference["residual_output"],
    }
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
        prefill_seq_len=prefill_seq_len,
        current_position=current_position,
        config=config,
        metadata=metadata,
        metadata_groups=metadata_groups,
        reference=reference,
        router_preview=router_preview,
        ffn_input_ids=ffn_input_ids,
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

    if prefill_seq_len % PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(
            f"TTNN prefill cache build seq_len must be a multiple of "
            f"{PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE}, got {prefill_seq_len}"
        )

    ttnn_outputs = _run_ttnn_decode_decoder_layer_slice(
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
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "ttnn.rms_norm(prefill_attn_norm)",
        "TtAttentionProjection.project_q_rank(prefill)",
        "ttnn.linear(wkv, prefill)",
        "host_rope_cache_prep(prefill)",
        "ttnn.rms_norm(decode_attn_norm)",
        "TtAttentionProjection.project_q_rank(decode)",
        "ttnn.linear(wkv, decode)",
        "host_rope_cache_prep(decode_position)",
        "host_append_current_kv_to_prefill_sliding_window",
        "TtSparsePrefillAttention(decode_one_token_host_gather)",
        "host_inverse_rope(decode_attention_output)",
        "host_grouped_wo_a",
        "ttnn.linear(wo_b)",
        "host_add(decode_hidden_state,attention_output_projected)",
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
        "prefill_cache": {
            "kv_cache_ready": _tensor_summary(ttnn_outputs["prefill_cache"]["kv_cache_ready"]),
            "sliding_window_cache": _tensor_summary(ttnn_outputs["prefill_cache"]["sliding_window_cache"]),
        },
        "decode_cache": {
            "attn_norm_output": _tensor_summary(ttnn_outputs["decode_cache"]["attn_norm_output"]),
            "q_rank_norm": _tensor_summary(ttnn_outputs["decode_cache"]["q_rank_norm"]),
            "q_output": _tensor_summary(ttnn_outputs["decode_cache"]["q_output"]),
            "kv_linear": _tensor_summary(ttnn_outputs["decode_cache"]["kv_linear"]),
            "kv_output": _tensor_summary(ttnn_outputs["decode_cache"]["kv_output"]),
            "q_decode": _tensor_summary(ttnn_outputs["decode_cache"]["q_decode"]),
            "kv_cache_ready": _tensor_summary(ttnn_outputs["decode_cache"]["kv_cache_ready"]),
        },
        "attention": {
            "attention_cache": _tensor_summary(ttnn_outputs["attention"]["attention_cache"]),
            "attention_output_rotary": _tensor_summary(ttnn_outputs["attention"]["attention_output_rotary"]),
            "attention_output": _tensor_summary(ttnn_outputs["attention"]["attention_output"]),
            "attention_output_flat": _tensor_summary(ttnn_outputs["attention"]["attention_output_flat"]),
            "output_rank": _tensor_summary(ttnn_outputs["attention"]["output_rank"]),
            "attention_output_projected": _tensor_summary(ttnn_outputs["attention"]["attention_output_projected"]),
        },
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
        "input_padding": ttnn_outputs["input_padding"],
    }
    result["ttnn_int"] = {
        "window_topk_idxs": _int_tensor_summary(ttnn_outputs["attention"]["window_topk_idxs"]),
        "compress_topk_idxs": _int_tensor_summary(ttnn_outputs["attention"]["compress_topk_idxs"]),
        "runtime_topk_idxs": _int_tensor_summary(ttnn_outputs["attention"]["runtime_topk_idxs"]),
    }
    accuracy = {
        "prefill_kv_cache_ready": _floating_accuracy_summary(
            reference["prefill_cache"]["kv_cache_ready"],
            ttnn_outputs["prefill_cache"]["kv_cache_ready"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "prefill_sliding_window_cache": _floating_accuracy_summary(
            reference["prefill_cache"]["sliding_window_cache"],
            ttnn_outputs["prefill_cache"]["sliding_window_cache"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "decode_attn_norm_output": _accuracy_summary(
            reference["decode_cache"]["attn_norm_output"],
            ttnn_outputs["decode_cache"]["attn_norm_output"],
            pcc_threshold=ffn_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "decode_q_rank_norm": _accuracy_summary(
            reference["decode_cache"]["q_rank_norm"],
            ttnn_outputs["decode_cache"]["q_rank_norm"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "decode_q_output": _accuracy_summary(
            reference["decode_cache"]["q_output"],
            ttnn_outputs["decode_cache"]["q_output"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "decode_kv_linear": _accuracy_summary(
            reference["decode_cache"]["kv_linear"],
            ttnn_outputs["decode_cache"]["kv_linear"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "decode_kv_output": _accuracy_summary(
            reference["decode_cache"]["kv_output"],
            ttnn_outputs["decode_cache"]["kv_output"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "decode_q_decode": _floating_accuracy_summary(
            reference["decode_cache"]["q_decode"],
            ttnn_outputs["decode_cache"]["q_decode"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "decode_kv_cache_ready": _floating_accuracy_summary(
            reference["decode_cache"]["kv_cache_ready"],
            ttnn_outputs["decode_cache"]["kv_cache_ready"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "attention_cache": _floating_accuracy_summary(
            reference["attention"]["attention_cache"],
            ttnn_outputs["attention"]["attention_cache"],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "attention_output_rotary": _floating_accuracy_summary(
            reference["attention"]["attention_output_rotary"],
            ttnn_outputs["attention"]["attention_output_rotary"],
            pcc_threshold=attention_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "attention_output": _floating_accuracy_summary(
            reference["attention"]["attention_output"],
            ttnn_outputs["attention"]["attention_output"],
            pcc_threshold=attention_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "attention_output_projected": _accuracy_summary(
            reference["attention"]["attention_output_projected"],
            ttnn_outputs["attention"]["attention_output_projected"],
            pcc_threshold=output_pcc,
            rtol=rtol,
            atol=attention_atol,
        ),
        "window_topk_idxs": _int_equality_summary(
            reference["attention"]["window_topk_idxs"],
            ttnn_outputs["attention"]["window_topk_idxs"],
        ),
        "compress_topk_idxs": _int_equality_summary(
            reference["attention"]["compress_topk_idxs"],
            ttnn_outputs["attention"]["compress_topk_idxs"],
        ),
        "runtime_topk_idxs": _int_equality_summary(
            reference["attention"]["runtime_topk_idxs"],
            ttnn_outputs["attention"]["runtime_topk_idxs"],
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
    }
    result["accuracy"] = accuracy
    result["router_match_stats"] = {
        "index_match_fraction": result["accuracy"]["router_indices"].get("match_fraction"),
        "index_mismatch_count": result["accuracy"]["router_indices"].get("mismatch_count"),
        "weights_pcc": result["accuracy"]["router_weights"].get("pcc"),
        "weights_max_abs": result["accuracy"]["router_weights"].get("max_abs"),
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_decode_decoder_layer_keys(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
) -> list[str]:
    return _unique_keys(
        [
            *layer_prefill_attention_runtime_keys(index, config=config, layer=layer),
            *layer_ffn_keys(index, layer=layer, expert=expert),
        ]
    )


def _prepare_decode_ffn_fanout(
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
    input_ids = _fanout_input_ids_for_activation(
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
    router_preview = _decode_fanout_preview(
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


def _fanout_input_ids_for_activation(
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
        raise ValueError("Hash-routed decode FFN fanout requires --expert to choose deterministic input ids")
    return deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=int(requested_expert), seq_len=seq_len)


def _decode_fanout_preview(
    *,
    requested_expert: int | None,
    input_ids: torch.Tensor | None,
    reference: Mapping[str, Any],
    config: DeepSeekV4FlashConfig,
) -> dict[str, Any]:
    router_weights = reference["router_weights"]
    router_indices = reference["router_indices"]
    return {
        "source": "torch_router_full_topk_on_post_attention_residual",
        "requested_expert": requested_expert,
        "input_id_anchor_expert": requested_expert if input_ids is not None else None,
        "topk": int(config.num_experts_per_tok),
        "activated_experts": _fanout_summary(router_weights, router_indices, full_topk=config.num_experts_per_tok),
        "per_expert_routes": {
            str(expert_id): _selection_summary(values["route"]) for expert_id, values in reference["per_expert"].items()
        },
    }


def layer_decode_decoder_layer_selector_keys(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> list[str]:
    keys = [
        *layer_prefill_attention_runtime_keys(index, config=config, layer=layer),
        f"layers.{layer}.ffn_norm.weight",
        f"layers.{layer}.ffn.gate.weight",
    ]
    for key in (f"layers.{layer}.ffn.gate.bias", f"layers.{layer}.ffn.gate.tid2eid"):
        if index.has_tensor(key):
            keys.append(key)
    return _unique_keys(keys)


def build_torch_decode_cache_prep_reference(
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    current_position: int,
) -> dict[str, torch.Tensor]:
    attention_reference = build_torch_attention_projection_reference(
        tensors,
        q_weights,
        config=config,
        layer=layer,
        activation=activation,
    )
    kv_reference = build_torch_kv_projection_reference(
        tensors,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
    )
    cache_reference = build_decode_cache_prep_from_projected(
        attention_reference["q_output"].to(torch.bfloat16),
        kv_reference["kv_output"].to(torch.bfloat16),
        config=config,
        layer=layer,
        current_position=current_position,
    )
    return {
        **attention_reference,
        **kv_reference,
        **cache_reference,
    }


def build_decode_cache_prep_from_projected(
    q_output: torch.Tensor,
    kv_output: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    current_position: int,
) -> dict[str, torch.Tensor]:
    if q_output.ndim != 4 or tuple(q_output.shape[:3]) != (1, 1, 1):
        raise ValueError(f"Expected decode q_output shape [1, 1, 1, width], got {tuple(q_output.shape)}")
    if kv_output.ndim != 4 or tuple(kv_output.shape[:3]) != (1, 1, 1):
        raise ValueError(f"Expected decode kv_output shape [1, 1, 1, width], got {tuple(kv_output.shape)}")
    head_dim = int(config.head_dim)
    q_width = int(config.num_attention_heads) * head_dim
    kv_width = int(config.num_key_value_heads) * head_dim
    if int(q_output.shape[-1]) != q_width:
        raise ValueError(f"Expected decode q width {q_width}, got {q_output.shape[-1]}")
    if int(kv_output.shape[-1]) != kv_width:
        raise ValueError(f"Expected decode kv width {kv_width}, got {kv_output.shape[-1]}")

    q_heads_pre_norm = q_output[:, 0].reshape(1, 1, int(config.num_attention_heads), head_dim).contiguous()
    q_heads = _unit_rms_norm(q_heads_pre_norm, eps=float(config.rms_norm_eps))
    kv_heads = kv_output[:, 0].reshape(1, 1, int(config.num_key_value_heads), head_dim).contiguous()
    if int(config.num_key_value_heads) != 1:
        raise ValueError(f"DeepSeek V4 Flash decode smoke expects one K/V head, got {config.num_key_value_heads}")
    return build_decode_cache_prep_from_splits(
        q_heads_pre_norm,
        q_heads,
        kv_heads,
        config=config,
        layer=layer,
        current_position=current_position,
    )


def build_decode_cache_prep_from_splits(
    q_heads_pre_norm: torch.Tensor,
    q_heads: torch.Tensor,
    kv_heads: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    current_position: int,
) -> dict[str, torch.Tensor]:
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    nope_dim = head_dim - rope_dim
    q_nope, q_rope = q_heads.split([nope_dim, rope_dim], dim=-1)
    kv_cache_projection = kv_heads[:, :, 0].contiguous()
    kv_nope, kv_rope = kv_cache_projection.split([nope_dim, rope_dim], dim=-1)
    freqs_cis = precompute_deepseek_v4_rope_frequencies(config, layer=layer, seq_len=current_position + 1)[
        current_position : current_position + 1
    ]
    q_rope_rotated = apply_deepseek_v4_rotary(q_rope.contiguous(), freqs_cis)
    kv_rope_rotated = apply_deepseek_v4_rotary(kv_rope.contiguous(), freqs_cis)
    q_decode = torch.cat([q_nope, q_rope_rotated], dim=-1).contiguous()
    kv_cache_ready = torch.cat([kv_nope, kv_rope_rotated], dim=-1).contiguous()
    return {
        "q_heads_pre_norm": q_heads_pre_norm.contiguous(),
        "q_heads": q_heads.contiguous(),
        "q_nope": q_nope.contiguous(),
        "q_rope": q_rope.contiguous(),
        "q_rope_rotated": q_rope_rotated,
        "q_decode": q_decode,
        "kv_heads": kv_heads.contiguous(),
        "kv_nope": kv_nope.contiguous(),
        "kv_rope": kv_rope.contiguous(),
        "kv_rope_rotated": kv_rope_rotated,
        "kv_cache_ready": kv_cache_ready,
        "rope_freqs_cis": freqs_cis,
    }


def build_torch_decode_attention_runtime_reference(
    prefill_cache_reference: Mapping[str, torch.Tensor],
    decode_cache_reference: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    attn_sink: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    current_position: int,
    indexer_weights: PrefillIndexerWeights | None = None,
) -> dict[str, torch.Tensor]:
    q_decode = decode_cache_reference["q_decode"].contiguous()
    batch_size, decode_tokens, num_heads, head_dim = q_decode.shape
    if (batch_size, decode_tokens) != (1, 1):
        raise ValueError(f"Expected decode q shape [1, 1, heads, dim], got {tuple(q_decode.shape)}")
    sliding_window_cache = _append_decode_kv_to_sliding_window(
        prefill_cache_reference["sliding_window_cache"],
        decode_cache_reference["kv_cache_ready"],
        sliding_window=int(config.sliding_window),
    )
    compressed_kv = _compressed_prefill_kv_from_cache(prefill_cache_reference, q_decode)
    indexer_reference = _decode_indexer_reference_from_prefill_cache(
        prefill_cache_reference,
        decode_cache_reference,
        indexer_weights,
        config=config,
        layer=layer,
        current_position=current_position,
        compressed_cache_length=int(compressed_kv.shape[1]),
    )
    attention_cache = torch.cat([sliding_window_cache, compressed_kv], dim=1).contiguous()
    window_topk_idxs = _decode_window_topk_indices(sliding_window_cache.shape[1], batch_size=batch_size)
    compress_topk_idxs = _decode_compress_topk_indices(
        indexer_reference,
        int(config.compress_ratios[layer]),
        batch_size=batch_size,
        current_position=current_position,
        compressed_cache_length=int(compressed_kv.shape[1]),
        offset=int(sliding_window_cache.shape[1]),
    )
    runtime_topk_idxs = torch.cat([window_topk_idxs, compress_topk_idxs], dim=-1).to(torch.int32)
    attention_output_rotary = sparse_attention(
        q_decode,
        attention_cache,
        attn_sink.float().contiguous(),
        runtime_topk_idxs,
        head_dim**-0.5,
    ).contiguous()
    local_only_attention_output_rotary = sparse_attention(
        q_decode,
        sliding_window_cache,
        attn_sink.float().contiguous(),
        window_topk_idxs,
        head_dim**-0.5,
    ).contiguous()
    attention_output = _inverse_attention_rope(
        attention_output_rotary,
        config=config,
        layer=layer,
        start_pos=current_position,
    )
    attention_output_flat = attention_output.reshape(batch_size, 1, num_heads * head_dim).to(torch.bfloat16)
    if q_weights.wo_a is None or q_weights.wo_b is None:
        raise ValueError("Output projection weights are required for decode attention runtime reference")
    output_rank = grouped_output_projection_a(attention_output_flat, q_weights.wo_a, o_groups=int(config.o_groups))
    attention_output_projected = F.linear(output_rank.float(), q_weights.wo_b.float()).unsqueeze(1)
    return {
        "prefill_sliding_window_cache": prefill_cache_reference["sliding_window_cache"].contiguous(),
        "current_kv_cache_ready": decode_cache_reference["kv_cache_ready"].contiguous(),
        "sliding_window_cache": sliding_window_cache,
        "compressed_kv": compressed_kv,
        **indexer_reference,
        "attention_cache": attention_cache,
        "window_topk_idxs": window_topk_idxs,
        "compress_topk_idxs": compress_topk_idxs,
        "runtime_topk_idxs": runtime_topk_idxs,
        "attention_output_rotary": attention_output_rotary,
        "local_only_attention_output_rotary": local_only_attention_output_rotary,
        "attention_output": attention_output,
        "attention_output_flat": attention_output_flat,
        "output_rank": output_rank,
        "attention_output_projected": attention_output_projected,
    }


def _compressed_prefill_kv_from_cache(
    prefill_cache: Mapping[str, torch.Tensor],
    q_decode: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(q_decode.shape[0])
    head_dim = int(q_decode.shape[-1])
    compressed_kv = prefill_cache.get("compressed_kv")
    if compressed_kv is None:
        return q_decode.new_empty(batch_size, 0, head_dim)
    if compressed_kv.ndim != 3:
        raise ValueError(
            f"compressed_kv must have shape [batch, cache_len, head_dim], got {tuple(compressed_kv.shape)}"
        )
    if int(compressed_kv.shape[0]) != batch_size or int(compressed_kv.shape[-1]) != head_dim:
        raise ValueError(
            "compressed_kv shape must match decode batch/head_dim, "
            f"got {tuple(compressed_kv.shape)} for decode {tuple(q_decode.shape)}"
        )
    return compressed_kv.contiguous()


def _decode_indexer_reference_from_prefill_cache(
    prefill_cache: Mapping[str, torch.Tensor],
    decode_cache: Mapping[str, torch.Tensor],
    indexer_weights: PrefillIndexerWeights | None,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    current_position: int,
    compressed_cache_length: int,
) -> dict[str, torch.Tensor]:
    q_decode = decode_cache["q_decode"].contiguous()
    batch_size = int(q_decode.shape[0])
    if compressed_cache_length <= 0 or indexer_weights is None:
        return _empty_indexer_reference(q_decode, config=config, seq_len=1)

    index_compressed_kv = prefill_cache.get("index_compressed_kv")
    if index_compressed_kv is None or index_compressed_kv.numel() == 0:
        return _empty_indexer_reference(q_decode, config=config, seq_len=1)
    if index_compressed_kv.ndim != 3 or int(index_compressed_kv.shape[0]) != batch_size:
        raise ValueError(
            "index_compressed_kv must have shape [batch, compressed_len, index_head_dim], "
            f"got {tuple(index_compressed_kv.shape)}"
        )

    q_rank = decode_cache["q_rank_norm"][:, 0].contiguous().to(torch.bfloat16)
    index_q = F.linear(q_rank.float(), indexer_weights.wq_b.float()).to(torch.bfloat16)
    index_q = index_q.reshape(batch_size, 1, int(config.index_n_heads), int(config.index_head_dim))
    index_q = _rotate_indexer_q(index_q, config=config, layer=layer, start_pos=current_position)

    attn_input = decode_cache["attn_norm_output"][:, 0].contiguous().to(torch.bfloat16)
    index_weights = F.linear(attn_input.float(), indexer_weights.weights_proj.float()).to(torch.bfloat16)
    index_weights = index_weights.float() * (int(config.index_head_dim) ** -0.5 * int(config.index_n_heads) ** -0.5)
    indexer_topk_idxs = indexer_topk(
        index_q,
        index_compressed_kv.contiguous(),
        index_weights,
        index_topk=int(config.index_topk),
        compress_ratio=int(config.compress_ratios[layer]),
        start_pos=current_position,
        offset=0,
    ).to(torch.int32)
    return {
        "index_q": index_q.contiguous(),
        "index_weights": index_weights.contiguous(),
        "index_compressed_kv": index_compressed_kv.contiguous(),
        "indexer_topk_idxs": indexer_topk_idxs.contiguous(),
    }


def _decode_compress_topk_indices(
    indexer_reference: Mapping[str, torch.Tensor],
    ratio: int,
    *,
    batch_size: int,
    current_position: int,
    compressed_cache_length: int,
    offset: int,
) -> torch.Tensor:
    if ratio <= 0 or compressed_cache_length <= 0:
        return torch.empty(batch_size, 1, 0, dtype=torch.int32)
    indexer_topk_idxs = indexer_reference["indexer_topk_idxs"]
    if indexer_topk_idxs.shape[-1] > 0:
        return _offset_valid_topk_indices(indexer_topk_idxs, offset=offset).to(torch.int32)
    topk = compress_topk_indices(
        ratio,
        batch_size,
        seq_len=1,
        start_pos=current_position,
        offset=offset,
    ).to(torch.int32)
    return topk[:, :, :compressed_cache_length]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real DeepSeek V4 Flash one-token decode decoder-layer TTNN smoke slice."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_DECODE_DECODER_LAYER)
    parser.add_argument(
        "--expert",
        type=int,
        default=None,
        help="Routed expert to materialize. Omit to choose the active expert for the decode token.",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_DECODE_DECODER_LAYER_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_DECODE_DECODER_LAYER_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cache-prep-pcc", type=float, default=0.99)
    parser.add_argument("--attention-pcc", type=float, default=0.99)
    parser.add_argument("--output-pcc", type=float, default=0.99)
    parser.add_argument("--residual-pcc", type=float, default=0.99)
    parser.add_argument("--ffn-norm-pcc", type=float, default=0.999)
    parser.add_argument("--router-pcc", type=float, default=0.99)
    parser.add_argument("--router-index-match", type=float, default=0.5)
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

    result = run_real_decode_decoder_layer_smoke(
        args.snapshot_dir,
        layer=args.layer,
        expert=args.expert,
        prefill_seq_len=args.prefill_seq_len,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        cache_prep_pcc=args.cache_prep_pcc,
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


def _run_ttnn_decode_decoder_layer_slice(
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    input_ids: torch.Tensor | None,
    prefill_activation: torch.Tensor,
    decode_activation: torch.Tensor,
    current_position: int,
    device_id: int,
    prefill_cache_outputs: Mapping[str, torch.Tensor] | None = None,
    indexer_weights: PrefillIndexerWeights | None = None,
) -> dict[str, Any]:
    cache_q_weights = _query_projection_weights_only(q_weights)
    if prefill_cache_outputs is None:
        prefill_cache_outputs = _run_ttnn_prefill_cache_prep(
            tensors,
            cache_q_weights,
            kv_weights,
            config=config,
            layer=layer,
            activation=prefill_activation,
            start_pos=0,
            device_id=device_id,
        )
    decode_cache_outputs = _run_ttnn_decode_cache_prep(
        tensors,
        cache_q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=decode_activation,
        current_position=current_position,
        device_id=device_id,
    )
    attention_outputs = _run_ttnn_decode_attention_from_cache_boundary(
        prefill_cache_outputs,
        decode_cache_outputs,
        q_weights,
        tensors[f"layers.{layer}.attn.attn_sink"],
        config=config,
        layer=layer,
        current_position=current_position,
        device_id=device_id,
        indexer_weights=indexer_weights,
    )
    post_attention_residual = _residual_add(decode_activation, attention_outputs["attention_output_projected"])
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
        "prefill_cache": prefill_cache_outputs,
        "decode_cache": decode_cache_outputs,
        "attention": attention_outputs,
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


def _run_ttnn_decode_cache_prep(
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    current_position: int,
    device_id: int,
) -> dict[str, torch.Tensor]:
    raw = _run_ttnn_prefill_cache_prep(
        tensors,
        q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=0,
        device_id=device_id,
    )
    decode_cache = build_decode_cache_prep_from_splits(
        raw["q_heads_pre_norm"],
        raw["q_heads"],
        raw["kv_heads"],
        config=config,
        layer=layer,
        current_position=current_position,
    )
    return {
        "attn_norm_output": raw["attn_norm_output"],
        "q_rank_norm": raw["q_rank_norm"],
        "q_output": raw["q_output"],
        "kv_linear": raw["kv_linear"],
        "kv_output": raw["kv_output"],
        **decode_cache,
    }


def _run_ttnn_decode_attention_from_cache_boundary(
    prefill_cache_outputs: Mapping[str, torch.Tensor],
    decode_cache_outputs: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    attn_sink: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    current_position: int,
    device_id: int,
    indexer_weights: PrefillIndexerWeights | None = None,
) -> dict[str, torch.Tensor]:
    import ttnn

    q_decode = decode_cache_outputs["q_decode"].contiguous()
    batch_size, decode_tokens, num_heads, head_dim = q_decode.shape
    if (batch_size, decode_tokens) != (1, 1):
        raise ValueError(f"Expected decode q shape [1, 1, heads, dim], got {tuple(q_decode.shape)}")
    sliding_window_cache = _append_decode_kv_to_sliding_window(
        prefill_cache_outputs["sliding_window_cache"],
        decode_cache_outputs["kv_cache_ready"],
        sliding_window=int(config.sliding_window),
    )
    compressed_kv = _compressed_prefill_kv_from_cache(prefill_cache_outputs, q_decode)
    indexer_outputs = _decode_indexer_reference_from_prefill_cache(
        prefill_cache_outputs,
        decode_cache_outputs,
        indexer_weights,
        config=config,
        layer=layer,
        current_position=current_position,
        compressed_cache_length=int(compressed_kv.shape[1]),
    )
    attention_cache = torch.cat([sliding_window_cache, compressed_kv], dim=1).contiguous()
    window_topk_idxs = _decode_window_topk_indices(sliding_window_cache.shape[1], batch_size=batch_size)
    compress_topk_idxs = _decode_compress_topk_indices(
        indexer_outputs,
        int(config.compress_ratios[layer]),
        batch_size=batch_size,
        current_position=current_position,
        compressed_cache_length=int(compressed_kv.shape[1]),
        offset=int(sliding_window_cache.shape[1]),
    )
    runtime_topk_idxs = torch.cat([window_topk_idxs, compress_topk_idxs], dim=-1).to(torch.int32)

    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        if q_weights.wo_a is None or q_weights.wo_b is None:
            raise ValueError("Output projection weights are required for TTNN decode attention runtime")
        projection_module = TtAttentionProjection(
            device=device,
            weights=q_weights,
            hidden_size=int(config.hidden_size),
            q_lora_rank=int(config.q_lora_rank),
            num_heads=int(config.num_attention_heads),
            head_dim=int(config.head_dim),
            norm_eps=float(config.rms_norm_eps),
            o_groups=int(config.o_groups),
            o_lora_rank=int(config.o_lora_rank),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q = ttnn.from_torch(
            q_decode.reshape(batch_size, 1, num_heads * head_dim).unsqueeze(1).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv = ttnn.from_torch(
            attention_cache.unsqueeze(1).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sparse_module = TtSparsePrefillAttention(
            device=device,
            num_heads=int(config.num_attention_heads),
            head_dim=int(config.head_dim),
            softmax_scale=int(config.head_dim) ** -0.5,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_attention_output_rotary = sparse_module(
            tt_q,
            tt_kv,
            attn_sink=attn_sink,
            topk_idxs=runtime_topk_idxs,
        )
        attention_output_rotary_flat = ttnn.to_torch(tt_attention_output_rotary).contiguous()[:, 0]
        attention_output_rotary = attention_output_rotary_flat.reshape(batch_size, 1, num_heads, head_dim)
        local_only_attention_output_rotary = sparse_attention(
            q_decode,
            sliding_window_cache,
            attn_sink.float().contiguous(),
            window_topk_idxs,
            head_dim**-0.5,
        ).contiguous()
        attention_output = _inverse_attention_rope(
            attention_output_rotary,
            config=config,
            layer=layer,
            start_pos=current_position,
        )
        attention_output_flat = attention_output.reshape(batch_size, 1, num_heads * head_dim).to(torch.bfloat16)
        output_rank = grouped_output_projection_a(attention_output_flat, q_weights.wo_a, o_groups=int(config.o_groups))
        tt_attention_output = ttnn.from_torch(
            attention_output_flat.unsqueeze(1).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_projected = projection_module.project_output(tt_attention_output)
        return {
            "prefill_sliding_window_cache": prefill_cache_outputs["sliding_window_cache"].contiguous(),
            "current_kv_cache_ready": decode_cache_outputs["kv_cache_ready"].contiguous(),
            "sliding_window_cache": sliding_window_cache,
            "compressed_kv": compressed_kv,
            **indexer_outputs,
            "attention_cache": attention_cache,
            "window_topk_idxs": window_topk_idxs,
            "compress_topk_idxs": compress_topk_idxs,
            "runtime_topk_idxs": runtime_topk_idxs,
            "attention_output_rotary": attention_output_rotary.contiguous(),
            "local_only_attention_output_rotary": local_only_attention_output_rotary.contiguous(),
            "attention_output": attention_output.contiguous(),
            "attention_output_flat": attention_output_flat.contiguous(),
            "output_rank": output_rank.contiguous(),
            "attention_output_projected": ttnn.to_torch(tt_projected).contiguous(),
        }
    finally:
        ttnn.close_device(device)


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    activated_experts: Sequence[int],
    requested_expert: int | None,
    prefill_seq_len: int,
    current_position: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    metadata_groups: Mapping[str, Sequence[TensorMetadata]],
    reference: Mapping[str, Any],
    router_preview: dict[str, Any],
    ffn_input_ids: torch.Tensor | None,
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    attention_metadata = metadata_groups["attention_runtime"]
    ffn_metadata = metadata_groups["ffn"]
    attention_payload = _attention_payload_byte_split(attention_metadata)
    ffn_payload = _ffn_fanout_payload_byte_split(ffn_metadata)
    total_payload = sum(item.nbytes for item in metadata)
    attention_reference = reference["attention"]
    decode_cache_reference = reference["decode_cache"]
    compress_ratio = int(config.compress_ratios[layer])
    primary_expert = int(activated_experts[0])
    return {
        "schema_version": REAL_DECODE_DECODER_LAYER_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "requested_expert": requested_expert,
        "expert": primary_expert,
        "activated_experts": [int(expert_id) for expert_id in activated_experts],
        "prefill_sequence_length": int(prefill_seq_len),
        "decode_tokens": 1,
        "current_position": int(current_position),
        "next_position": int(current_position + 1),
        "model": {
            "hidden_size": config.hidden_size,
            "q_lora_rank": config.q_lora_rank,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "o_groups": config.o_groups,
            "o_lora_rank": config.o_lora_rank,
            "compress_ratio": compress_ratio,
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
                "prefill cache build -> one-token attn_norm/Q/KV decode projection -> sparse attention over "
                "prefill sliding-window KV plus current KV -> attention residual -> FFN residual"
            ),
            "layer_choice": "layer 3 is the default because the real prefill decoder and FFN helpers are proven there",
            "compressed_decode_cache": (
                "not exercised in this first layer-3 decode step because current_position + 1 is below "
                "the layer compress_ratio"
            ),
            "full_expert_fanout": (
                "enabled for the decode-token FFN after attention; the prefill cache build remains cache-only and "
                "does not materialize a broad prefill FFN fanout"
            ),
            "embeddings_logits_vllm_evals": "excluded",
        },
        "fanout_scope": {
            "decode_ffn_full_expert_fanout": "enabled for every router top-k expert selected by the decode token",
            "prefill_ffn_full_expert_fanout": "not materialized in this decode-layer smoke",
            "activated_expert_count": len(activated_experts),
            "activated_expert_ids": [int(expert_id) for expert_id in activated_experts],
            "topk": int(config.num_experts_per_tok),
            "topk_prefix_limit": None,
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
        "cache_sizes": {
            "prefill_attention_tokens": int(prefill_seq_len),
            "current_token_cache_tokens": 1,
            "sliding_window_cache_before_decode": int(reference["prefill_cache"]["sliding_window_cache"].shape[1]),
            "sliding_window_cache_after_decode": int(attention_reference["sliding_window_cache"].shape[1]),
            "compressed_cache_length": int(attention_reference["compressed_kv"].shape[1]),
            "attention_cache_length": int(attention_reference["attention_cache"].shape[1]),
            "runtime_topk_width": int(attention_reference["runtime_topk_idxs"].shape[-1]),
            "window_topk_valid_count": int((attention_reference["window_topk_idxs"] >= 0).sum().item()),
            "compress_topk_valid_count": int((attention_reference["compress_topk_idxs"] >= 0).sum().item()),
        },
        "output_shapes": {
            "prefill_input_hidden_states": list(reference["prefill_input_hidden_states"].shape),
            "decode_input_hidden_states": list(reference["decode_input_hidden_states"].shape),
            "decode_attn_norm": list(decode_cache_reference["attn_norm_output"].shape),
            "decode_q": list(decode_cache_reference["q_decode"].shape),
            "decode_kv_cache_ready": list(decode_cache_reference["kv_cache_ready"].shape),
            "attention_cache": list(attention_reference["attention_cache"].shape),
            "attention_output": list(attention_reference["attention_output"].shape),
            "attention_output_projected": list(attention_reference["attention_output_projected"].shape),
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
        "selected_expert_info": router_preview,
        "ffn_fanout_info": router_preview,
        "sparse_attention_inputs": {
            "window_topk_idxs": _int_tensor_summary(attention_reference["window_topk_idxs"]),
            "compress_topk_idxs": _int_tensor_summary(attention_reference["compress_topk_idxs"]),
            "runtime_topk_idxs": _int_tensor_summary(attention_reference["runtime_topk_idxs"]),
            "window_topk_source": "host-visible decode window indices over valid prefill KV plus current KV",
            "compressed_topk_source": "empty for this first layer-3 decode step",
            "attention_cache_source": "prefill sliding_window_cache concatenated with current decode kv_cache_ready",
            "current_position_included": bool(
                int(attention_reference["window_topk_idxs"].max().item())
                == int(attention_reference["attention_cache"].shape[1] - 1)
            ),
        },
        "host_boundaries": [
            {
                "name": "projection_fp8_decode_to_bf16",
                "location": "before TTNN attention projection modules",
                "description": "real FP8 Q, K/V, wo_a, and wo_b weights are decoded on host to BF16",
            },
            {
                "name": "prefill_cache_readback",
                "location": "after real Q/KV prefill cache prep",
                "description": "prefill kv_cache_ready and sliding_window_cache are copied to host",
            },
            {
                "name": "decode_rope_cache_prep_host",
                "location": "after one-token TTNN Q/KV projection split",
                "description": "position-specific RoPE and decode cache-ready concat run on host",
            },
            {
                "name": "decode_attention_cache_host_append",
                "location": "before sparse attention",
                "description": "current token KV is appended to the host-visible prefill sliding-window cache",
            },
            {
                "name": "sparse_attention_host_fallback",
                "location": "inside TtSparsePrefillAttention",
                "description": "indexed gather, attention-sink softmax, and weighted reduction currently run on host",
            },
            {
                "name": "inverse_rope_host",
                "location": "after sparse attention",
                "description": "decode attention output RoPE dimensions are inverse-rotated on host",
            },
            {
                "name": "grouped_wo_a_host",
                "location": "inside output projection",
                "description": "grouped wo_a projection runs on host before TTNN wo_b",
            },
            {
                "name": "attention_residual_host_add",
                "location": "decoder layer attention residual",
                "description": "attention projection output is added to the decode hidden state on host",
            },
            {
                "name": "activated_expert_slice_selection",
                "location": "before loading routed expert weights",
                "description": "all decode-token router top-k routed experts are selected before loading weights",
            },
            {
                "name": "routed_fp4_decode_to_bf16",
                "location": "before TtRoutedExpertMLP",
                "description": (
                    "packed-FP4 routed expert weights and scales are decoded on host to BF16 for each "
                    "activated expert"
                ),
            },
            {
                "name": "router_topk",
                "location": "TtRouter",
                "description": "router scores leave device for host DeepSeek top-k/hash selection",
            },
            {
                "name": "activated_expert_gather",
                "location": "between router and routed expert",
                "description": "activated expert token activations and route weights are gathered on host per expert",
            },
            {
                "name": "activated_expert_scatter_add",
                "location": "after TtRoutedExpertMLP",
                "description": "all activated expert contributions are scattered and accumulated on host",
            },
            {
                "name": "ffn_host_combine",
                "location": "after routed and shared experts",
                "description": "routed and shared expert contributions are added on host",
            },
            {
                "name": "ffn_residual_host_add",
                "location": "decoder layer FFN residual",
                "description": "combined FFN contribution is added to the post-attention residual on host",
            },
        ],
        "reference_ops": [
            "torch.rms_norm_reference(prefill_attn_norm)",
            "torch.real_prefill_cache_prep_reference",
            "torch.rms_norm_reference(decode_attn_norm)",
            "torch.real_decode_q_kv_projection_reference",
            "host_append_current_kv_to_prefill_sliding_window",
            "torch.sparse_attention_decode_reference",
            "host_add(decode_hidden_state,attention_output_projected)",
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
        "reference": {
            "decode_cache": {
                "attn_norm_output": _tensor_summary(decode_cache_reference["attn_norm_output"]),
                "q_decode": _tensor_summary(decode_cache_reference["q_decode"]),
                "kv_cache_ready": _tensor_summary(decode_cache_reference["kv_cache_ready"]),
            },
            "attention": {
                "attention_cache": _tensor_summary(attention_reference["attention_cache"]),
                "attention_output": _tensor_summary(attention_reference["attention_output"]),
                "attention_output_projected": _tensor_summary(attention_reference["attention_output_projected"]),
            },
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
        "inputs": {
            "prefill_hidden_states": _tensor_summary(reference["prefill_input_hidden_states"]),
            "decode_hidden_state": _tensor_summary(reference["decode_input_hidden_states"]),
            "decode_ffn_input_ids": _tensor_summary(ffn_input_ids),
        },
        "router_match_stats": {},
        "ttnn": {},
        "ttnn_int": {},
        "accuracy": {},
        "passed": False,
    }


def _append_decode_kv_to_sliding_window(
    prefill_sliding_window_cache: torch.Tensor,
    current_kv_cache_ready: torch.Tensor,
    *,
    sliding_window: int,
) -> torch.Tensor:
    if prefill_sliding_window_cache.ndim != 3:
        raise ValueError(
            "prefill_sliding_window_cache must have shape [batch, cache_len, head_dim], "
            f"got {tuple(prefill_sliding_window_cache.shape)}"
        )
    if tuple(current_kv_cache_ready.shape[:2]) != (prefill_sliding_window_cache.shape[0], 1):
        raise ValueError(
            "current_kv_cache_ready must have shape [batch, 1, head_dim], " f"got {tuple(current_kv_cache_ready.shape)}"
        )
    if current_kv_cache_ready.shape[-1] != prefill_sliding_window_cache.shape[-1]:
        raise ValueError(
            f"current KV head_dim {current_kv_cache_ready.shape[-1]} must match "
            f"prefill cache head_dim {prefill_sliding_window_cache.shape[-1]}"
        )
    combined = torch.cat([prefill_sliding_window_cache, current_kv_cache_ready], dim=1).contiguous()
    if combined.shape[1] > sliding_window:
        return combined[:, -sliding_window:].contiguous()
    return combined


def _decode_window_topk_indices(cache_len: int, *, batch_size: int) -> torch.Tensor:
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    return torch.arange(cache_len, dtype=torch.int32).reshape(1, 1, cache_len).expand(batch_size, 1, cache_len)


def _validate_runtime_config(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    prefill_seq_len: int,
) -> None:
    if layer >= len(config.compress_ratios):
        raise ValueError(f"layer {layer} is outside compress_ratios length {len(config.compress_ratios)}")
    if int(config.num_key_value_heads) != 1:
        raise ValueError(f"Expected one K/V head, got {config.num_key_value_heads}")
    if prefill_seq_len + 1 > int(config.sliding_window):
        raise ValueError(
            f"This first decode smoke requires prefill_seq_len + 1 <= sliding_window {config.sliding_window}, "
            f"got {prefill_seq_len + 1}"
        )
    compress_ratio = int(config.compress_ratios[layer])
    if compress_ratio > 0 and prefill_seq_len + 1 >= compress_ratio:
        raise ValueError(
            "This first real decode decoder-layer smoke only supports the pre-compressed layer-3 stepping stone; "
            f"got prefill_seq_len + 1 = {prefill_seq_len + 1} and compress_ratio = {compress_ratio}"
        )


def _validate_smoke_args(
    *,
    layer: int,
    expert: int | None,
    prefill_seq_len: int,
    max_tensors: int,
    max_bytes: int,
    pcc_thresholds: Mapping[str, float],
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if expert is not None and expert < 0:
        raise ValueError(f"expert must be non-negative when provided, got {expert}")
    if prefill_seq_len <= 0:
        raise ValueError(f"prefill_seq_len must be positive, got {prefill_seq_len}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


if __name__ == "__main__":
    main()
