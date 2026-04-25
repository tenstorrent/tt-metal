# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm, v4_router
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_EXPERT_MLP_EXPERT,
    DEFAULT_LAYER_EXPERT_MLP_LAYER,
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_shared_expert_mlp_keys,
)
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights, validate_real_expert_slice
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import (
    ROUTED_MOE_TTNN_TILE_MULTIPLE,
    build_torch_selected_expert_reference,
    deterministic_input_ids_for_expert,
    deterministic_routed_activation,
    layer_routed_moe_keys,
    selected_expert_route,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import (
    SHARED_EXPERT_FP8_BLOCK_SIZE,
    build_torch_shared_expert_reference,
    decode_real_shared_expert_weights,
    validate_real_shared_expert_slice,
)

REAL_FFN_SMOKE_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_FFN_MAX_TENSORS = 15
DEFAULT_FFN_MAX_BYTES = 64 * 1024 * 1024


def run_real_ffn_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_EXPERT_MLP_LAYER,
    expert: int = DEFAULT_LAYER_EXPERT_MLP_EXPERT,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_FFN_MAX_TENSORS,
    max_bytes: int = DEFAULT_FFN_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    norm_pcc: float = 0.999,
    router_pcc: float = 0.99,
    router_index_match: float = 0.8,
    routed_pcc: float = 0.99,
    shared_pcc: float = 0.99,
    combined_pcc: float = 0.99,
    residual_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Run one real DeepSeek V4 Flash FFN composition slice with one routed expert and shared expert."""

    _validate_smoke_args(
        layer=layer,
        expert=expert,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        norm_pcc=norm_pcc,
        router_pcc=router_pcc,
        router_index_match=router_index_match,
        routed_pcc=routed_pcc,
        shared_pcc=shared_pcc,
        combined_pcc=combined_pcc,
        residual_pcc=residual_pcc,
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_real_ffn_slice(
        snapshot_dir,
        layer=layer,
        expert=expert,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    validate_real_ffn_slice(tensors, config=config, layer=layer, expert=expert)
    routed_weights = decode_real_expert_weights(tensors, config=config, layer=layer, expert=expert)
    shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=layer)

    activation = deterministic_routed_activation(
        hidden_size=config.hidden_size,
        seq_len=seq_len,
        gate_weight=tensors[f"layers.{layer}.ffn.gate.weight"],
        expert=expert,
    )
    reference = build_torch_ffn_reference(
        tensors,
        routed_weights,
        shared_weights,
        config=config,
        layer=layer,
        expert=expert,
        activation=activation,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        expert=expert,
        seq_len=seq_len,
        config=config,
        metadata=metadata,
        tensors=tensors,
        routed_weights=routed_weights,
        shared_weights=shared_weights,
        activation=activation,
        reference=reference,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    if seq_len % ROUTED_MOE_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN smoke seq_len must be a multiple of {ROUTED_MOE_TTNN_TILE_MULTIPLE}, got {seq_len}")

    ttnn_outputs = _run_ttnn_ffn_slice(
        tensors,
        routed_weights,
        shared_weights,
        config=config,
        layer=layer,
        expert=expert,
        activation=activation,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "ttnn.rms_norm(ffn_norm)",
        "TtRouter(ttnn.linear+host_topk)",
        "host_gather_selected_expert_tokens",
        "TtRoutedExpertMLP",
        "host_scatter_selected_expert_output",
        "TtSharedExpertMLP",
        "host_add(routed_expert_output,shared_expert_output)",
        "host_add(residual_input,combined_ffn_output)",
    ]
    result["ttnn"] = {
        "rms_norm": _tensor_summary(ttnn_outputs["norm_output"]),
        "router_weights": _tensor_summary(ttnn_outputs["router_weights"]),
        "router_indices": _tensor_summary(ttnn_outputs["router_indices"]),
        "selected_route": _selection_summary(ttnn_outputs["selected_route"]),
        "selected_expert_output": _tensor_summary(ttnn_outputs["selected_expert_output"]),
        "routed_output": _tensor_summary(ttnn_outputs["routed_output"]),
        "shared_output": _tensor_summary(ttnn_outputs["shared_output"]),
        "combined_output": _tensor_summary(ttnn_outputs["combined_output"]),
        "residual_output": _tensor_summary(ttnn_outputs["residual_output"]),
        "expert_padding": ttnn_outputs["expert_padding"],
    }
    result["accuracy"] = {
        "rms_norm": _accuracy_summary(
            reference["norm_output"],
            ttnn_outputs["norm_output"],
            pcc_threshold=norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "router_weights": _accuracy_summary(
            reference["router_weights"],
            ttnn_outputs["router_weights"],
            pcc_threshold=router_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "router_indices": _index_accuracy_summary(
            reference["router_indices"],
            ttnn_outputs["router_indices"],
            match_threshold=router_index_match,
        ),
        "routed_output": _accuracy_summary(
            reference["routed_output"],
            ttnn_outputs["routed_output"],
            pcc_threshold=routed_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "shared_output": _accuracy_summary(
            reference["shared_output"],
            ttnn_outputs["shared_output"],
            pcc_threshold=shared_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "combined_output": _accuracy_summary(
            reference["combined_output"],
            ttnn_outputs["combined_output"],
            pcc_threshold=combined_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "residual_output": _accuracy_summary(
            reference["residual_output"],
            ttnn_outputs["residual_output"],
            pcc_threshold=residual_pcc,
            rtol=rtol,
            atol=atol,
        ),
    }
    result["router_match_stats"] = {
        "index_match_fraction": result["accuracy"]["router_indices"].get("match_fraction"),
        "index_mismatch_count": result["accuracy"]["router_indices"].get("mismatch_count"),
        "weights_pcc": result["accuracy"]["router_weights"].get("pcc"),
        "weights_max_abs": result["accuracy"]["router_weights"].get("max_abs"),
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def load_real_ffn_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    expert: int,
    max_tensors: int = DEFAULT_FFN_MAX_TENSORS,
    max_bytes: int = DEFAULT_FFN_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_ffn_keys(index, layer=layer, expert=expert)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def layer_ffn_keys(index: RealCheckpointTensorIndex, *, layer: int, expert: int) -> list[str]:
    return [
        *layer_routed_moe_keys(index, layer=layer, expert=expert),
        *layer_shared_expert_mlp_keys(index, layer=layer),
    ]


def validate_real_ffn_slice(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
) -> None:
    _validate_router_norm_tensors(tensors, config=config, layer=layer)
    validate_real_expert_slice(tensors, config=config, layer=layer, expert=expert)
    validate_real_shared_expert_slice(tensors, config=config, layer=layer)


def build_torch_ffn_reference(
    tensors: Mapping[str, torch.Tensor],
    routed_weights: Mapping[str, torch.Tensor],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
    activation: torch.Tensor,
) -> dict[str, Any]:
    _validate_activation(activation, hidden_size=config.hidden_size)
    prefix = f"layers.{layer}"
    norm_weight = tensors[f"{prefix}.ffn_norm.weight"].to(torch.bfloat16)
    gate_weight = tensors[f"{prefix}.ffn.gate.weight"].to(torch.bfloat16)
    bias = tensors.get(f"{prefix}.ffn.gate.bias")
    tid2eid = tensors.get(f"{prefix}.ffn.gate.tid2eid")
    input_ids = (
        deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=expert, seq_len=activation.shape[-2])
        if tid2eid is not None
        else None
    )

    norm_output = rms_norm(activation[:, 0], norm_weight, eps=config.rms_norm_eps).unsqueeze(1).to(torch.bfloat16)
    router_weights, router_indices = v4_router(
        norm_output[:, 0],
        gate_weight,
        topk=config.num_experts_per_tok,
        route_scale=config.routed_scaling_factor,
        scoring_func=config.scoring_func,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
    )
    selected_route = selected_expert_route(router_weights, router_indices, expert=expert)
    routed_outputs = build_torch_selected_expert_reference(
        routed_weights,
        config=config,
        hidden_states=norm_output,
        selected_route=selected_route,
    )
    shared_output = build_torch_shared_expert_reference(
        shared_weights,
        config=config,
        activation=norm_output,
    )
    combined_output = (routed_outputs["expert_scattered_output"].float() + shared_output.float()).to(norm_output.dtype)
    residual_output = (activation.float() + combined_output.float()).to(activation.dtype)
    return {
        "norm_output": norm_output,
        "router_weights": router_weights,
        "router_indices": router_indices,
        "input_ids": input_ids,
        "selected_route": selected_route,
        "selected_expert_input": routed_outputs["selected_expert_input"],
        "selected_expert_output": routed_outputs["selected_expert_output"],
        "routed_output": routed_outputs["expert_scattered_output"],
        "shared_output": shared_output,
        "combined_output": combined_output,
        "residual_output": residual_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny DeepSeek V4 Flash real FFN TTNN composition slice.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_EXPERT_MLP_LAYER)
    parser.add_argument("--expert", type=int, default=DEFAULT_LAYER_EXPERT_MLP_EXPERT)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_FFN_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_FFN_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--norm-pcc", type=float, default=0.999)
    parser.add_argument("--router-pcc", type=float, default=0.99)
    parser.add_argument("--router-index-match", type=float, default=0.8)
    parser.add_argument("--routed-pcc", type=float, default=0.99)
    parser.add_argument("--shared-pcc", type=float, default=0.99)
    parser.add_argument("--combined-pcc", type=float, default=0.99)
    parser.add_argument("--residual-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_ffn_smoke(
        args.snapshot_dir,
        layer=args.layer,
        expert=args.expert,
        seq_len=args.seq_len,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        norm_pcc=args.norm_pcc,
        router_pcc=args.router_pcc,
        router_index_match=args.router_index_match,
        routed_pcc=args.routed_pcc,
        shared_pcc=args.shared_pcc,
        combined_pcc=args.combined_pcc,
        residual_pcc=args.residual_pcc,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_ffn_slice(
    tensors: Mapping[str, torch.Tensor],
    routed_weights: Mapping[str, torch.Tensor],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
    activation: torch.Tensor,
    device_id: int,
) -> dict[str, Any]:
    import ttnn
    from models.demos.deepseek_v4_flash.real_routed_moe_smoke import _pad_selected_expert_inputs
    from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP
    from models.demos.deepseek_v4_flash.ttnn_router import TtRouter
    from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP

    prefix = f"layers.{layer}"
    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_input = ttnn.from_torch(
            activation,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_norm_weight = ttnn.from_torch(
            tensors[f"{prefix}.ffn_norm.weight"].contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_norm = ttnn.rms_norm(
            tt_input,
            weight=tt_norm_weight,
            epsilon=config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tid2eid = tensors.get(f"{prefix}.ffn.gate.tid2eid")
        input_ids = (
            deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=expert, seq_len=activation.shape[-2])
            if tid2eid is not None
            else None
        )
        router = TtRouter(
            device=device,
            gate_weight=tensors[f"{prefix}.ffn.gate.weight"],
            bias=tensors.get(f"{prefix}.ffn.gate.bias"),
            tid2eid=tid2eid,
            topk=config.num_experts_per_tok,
            route_scale=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        router_weights, router_indices = router(tt_norm, input_ids=input_ids)
        selected_route = selected_expert_route(router_weights, router_indices, expert=expert)

        norm_output = ttnn.to_torch(tt_norm).contiguous()
        selected_input = norm_output[:, :, selected_route.token_indices, :].contiguous()
        padded_input, padded_route_weight, padding = _pad_selected_expert_inputs(
            selected_input,
            selected_route.route_weight,
            intermediate_size=routed_weights["w1"].shape[0],
        )
        tt_selected_input = ttnn.from_torch(
            padded_input,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_route_weight = ttnn.from_torch(
            padded_route_weight,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routed_module = TtRoutedExpertMLP(
            device=device,
            w1=routed_weights["w1"],
            w2=routed_weights["w2"],
            w3=routed_weights["w3"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            swiglu_limit=config.swiglu_limit,
        )
        padded_routed_output = ttnn.to_torch(
            routed_module(tt_selected_input, route_weight=tt_route_weight)
        ).contiguous()
        selected_output = padded_routed_output[:, :, : selected_route.selected_token_count, :].contiguous()
        routed_output = torch.zeros_like(norm_output)
        routed_output[:, :, selected_route.token_indices, :] = selected_output

        shared_module = TtSharedExpertMLP(
            device=device,
            w1=shared_weights["w1"],
            w2=shared_weights["w2"],
            w3=shared_weights["w3"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            swiglu_limit=config.swiglu_limit,
        )
        shared_output = ttnn.to_torch(shared_module(tt_norm)).contiguous()
        combined_output = (routed_output.float() + shared_output.float()).to(norm_output.dtype)
        residual_output = (activation.float() + combined_output.float()).to(activation.dtype)
        return {
            "norm_output": norm_output,
            "router_weights": router_weights.contiguous(),
            "router_indices": router_indices.contiguous(),
            "selected_route": selected_route,
            "selected_expert_output": selected_output,
            "routed_output": routed_output,
            "shared_output": shared_output,
            "combined_output": combined_output,
            "residual_output": residual_output,
            "expert_padding": padding,
        }
    finally:
        ttnn.close_device(device)


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    expert: int,
    seq_len: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    tensors: Mapping[str, torch.Tensor],
    routed_weights: Mapping[str, torch.Tensor],
    shared_weights: Mapping[str, torch.Tensor],
    activation: torch.Tensor,
    reference: dict[str, Any],
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = _payload_byte_split(metadata)
    return {
        "schema_version": REAL_FFN_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "expert": int(expert),
        "sequence_length": int(seq_len),
        "model": {
            "hidden_size": config.hidden_size,
            "moe_intermediate_size": config.moe_intermediate_size,
            "n_routed_experts": config.n_routed_experts,
            "n_shared_experts": config.n_shared_experts,
            "shared_intermediate_size": int(config.moe_intermediate_size) * int(config.n_shared_experts),
            "num_experts_per_tok": config.num_experts_per_tok,
            "scoring_func": config.scoring_func,
            "routed_scaling_factor": config.routed_scaling_factor,
            "rms_norm_eps": config.rms_norm_eps,
            "swiglu_limit": config.swiglu_limit,
            "expert_dtype": config.expert_dtype,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "residual_semantics": "residual_output = activation + (selected_routed_expert_output + shared_expert_output)",
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "payload_bytes": payload_bytes,
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": payload_bytes["total"],
        },
        "selected_expert_token_count": reference["selected_route"].selected_token_count,
        "expert_format": {
            "routed": {
                "format": "PACKED_FP4",
                "decoded_tensors": {
                    projection: _tensor_summary(weight) for projection, weight in routed_weights.items()
                },
            },
            "shared": {
                "format": "FP8_E4M3_WEIGHT_UE8M0_128x128_SCALE_OR_DIRECT_FLOAT",
                "weight_block_size": list(SHARED_EXPERT_FP8_BLOCK_SIZE),
                "source_formats": _shared_source_format_summary(tensors, layer=layer),
                "decoded_tensors": {
                    projection: _tensor_summary(weight) for projection, weight in shared_weights.items()
                },
            },
        },
        "host_boundaries": [
            {
                "name": "routed_fp4_decode_to_bf16",
                "location": "before TtRoutedExpertMLP",
                "description": "packed-FP4 routed expert weights and scales are decoded on host to BF16",
            },
            {
                "name": "shared_fp8_decode_to_bf16",
                "location": "before TtSharedExpertMLP",
                "description": "FP8 E4M3 shared expert weights and UE8M0 block scales are decoded on host to BF16",
            },
            {
                "name": "router_topk",
                "location": "TtRouter",
                "description": "router scores leave device for host DeepSeek top-k/hash selection",
            },
            {
                "name": "normalized_hidden_readback",
                "location": "between router and routed expert",
                "description": "normalized hidden states are copied to host to gather tokens for one selected expert",
            },
            {
                "name": "selected_expert_gather",
                "location": "between router and routed expert",
                "description": "selected expert token activations and route weights are gathered on host",
            },
            {
                "name": "selected_expert_tile_padding",
                "location": "before TtRoutedExpertMLP",
                "description": "selected-token batch is padded to a TTNN tile multiple, with zero route weight on padding",
            },
            {
                "name": "selected_expert_scatter",
                "location": "after TtRoutedExpertMLP",
                "description": "single-expert contribution is scattered back to full sequence shape on host",
            },
            {
                "name": "shared_expert_readback",
                "location": "after TtSharedExpertMLP",
                "description": "shared expert output is copied to host for this smoke-test combine and comparison",
            },
            {
                "name": "ffn_host_combine",
                "location": "after routed and shared experts",
                "description": "routed and shared expert contributions are added on host",
            },
            {
                "name": "residual_host_add",
                "location": "FFN block output",
                "description": "the FFN contribution is added to the original residual/input on host",
            },
        ],
        "reference_ops": [
            "torch.rms_norm_reference(ffn_norm)",
            "torch.router_reference",
            "host_gather_selected_expert_tokens",
            "torch.routed_swiglu_expert_reference",
            "host_scatter_selected_expert_output",
            "torch.shared_swiglu_expert_reference",
            "host_add(routed_expert_output,shared_expert_output)",
            "host_add(residual_input,combined_ffn_output)",
        ],
        "ttnn_ops": [],
        "inputs": {
            "activation": _tensor_summary(activation),
            "input_ids": _tensor_summary(reference["input_ids"]),
        },
        "reference": {
            "rms_norm": _tensor_summary(reference["norm_output"]),
            "router_weights": _tensor_summary(reference["router_weights"]),
            "router_indices": _tensor_summary(reference["router_indices"]),
            "selected_route": _selection_summary(reference["selected_route"]),
            "selected_expert_input": _tensor_summary(reference["selected_expert_input"]),
            "selected_expert_output": _tensor_summary(reference["selected_expert_output"]),
            "routed_output": _tensor_summary(reference["routed_output"]),
            "shared_output": _tensor_summary(reference["shared_output"]),
            "combined_output": _tensor_summary(reference["combined_output"]),
            "residual_output": _tensor_summary(reference["residual_output"]),
        },
        "router_match_stats": {},
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _validate_router_norm_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> None:
    prefix = f"layers.{layer}"
    expected_shapes = {
        f"{prefix}.ffn_norm.weight": (config.hidden_size,),
        f"{prefix}.ffn.gate.weight": (config.n_routed_experts, config.hidden_size),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in tensors:
            raise KeyError(f"Missing required real FFN tensor {key!r}")
        if tuple(tensors[key].shape) != expected_shape:
            raise ValueError(f"Expected {key} shape {expected_shape}, got {tuple(tensors[key].shape)}")

    bias_key = f"{prefix}.ffn.gate.bias"
    tid2eid_key = f"{prefix}.ffn.gate.tid2eid"
    has_bias = bias_key in tensors
    has_tid2eid = tid2eid_key in tensors
    if has_bias == has_tid2eid:
        raise ValueError(f"Expected exactly one of {bias_key!r} or {tid2eid_key!r}")
    if has_bias and tuple(tensors[bias_key].shape) != (config.n_routed_experts,):
        raise ValueError(
            f"Expected {bias_key} shape {(config.n_routed_experts,)}, got {tuple(tensors[bias_key].shape)}"
        )
    if has_tid2eid:
        expected_tid2eid_shape = (config.vocab_size, config.num_experts_per_tok)
        if tuple(tensors[tid2eid_key].shape) != expected_tid2eid_shape:
            raise ValueError(
                f"Expected {tid2eid_key} shape {expected_tid2eid_shape}, got {tuple(tensors[tid2eid_key].shape)}"
            )


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {"norm": 0, "router": 0, "routed_expert": 0, "shared_expert": 0}
    for item in metadata:
        if ".ffn_norm." in item.canonical_key:
            split["norm"] += item.nbytes
        elif ".ffn.gate." in item.canonical_key:
            split["router"] += item.nbytes
        elif ".ffn.experts." in item.canonical_key:
            split["routed_expert"] += item.nbytes
        elif ".ffn.shared_experts." in item.canonical_key:
            split["shared_expert"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in FFN slice: {item.canonical_key}")
    split["total"] = sum(split.values())
    return split


def _metadata_summary(item: TensorMetadata) -> dict[str, Any]:
    return {
        "canonical_key": item.canonical_key,
        "source_key": item.source_key,
        "shard": item.shard_name,
        "dtype": item.dtype,
        "shape": list(item.shape),
        "nbytes": item.nbytes,
    }


def _shared_source_format_summary(tensors: Mapping[str, torch.Tensor], *, layer: int) -> dict[str, Any]:
    formats = {}
    for projection in ("w1", "w2", "w3"):
        prefix = f"layers.{layer}.ffn.shared_experts.{projection}"
        weight = tensors[f"{prefix}.weight"]
        scale = tensors[f"{prefix}.scale"]
        if weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu:
            source_format = "FP8_E4M3_WEIGHT_UE8M0_128x128_SCALE"
        elif weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            source_format = "DIRECT_FLOAT_WEIGHT_SCALE_IGNORED"
        else:
            source_format = "UNSUPPORTED"
        formats[projection] = {
            "format": source_format,
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "scale_dtype": str(scale.dtype),
            "scale_shape": list(scale.shape),
        }
    return formats


def _selection_summary(selection) -> dict[str, Any]:
    return {
        "expert": selection.expert,
        "selected_token_count": selection.selected_token_count,
        "total_tokens": selection.total_tokens,
        "hit_count": selection.hit_count,
        "token_indices_preview": [int(value) for value in selection.token_indices[:16].tolist()],
        "topk_slot_histogram": {str(key): value for key, value in selection.topk_slot_histogram.items()},
        "route_weight": _tensor_summary(selection.route_weight),
        "full_route_weight": _tensor_summary(selection.full_route_weight),
    }


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": 0}
    tensor_float = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor_float.min().item()),
        "max": float(tensor_float.max().item()),
        "mean": float(tensor_float.mean().item()),
    }


def _accuracy_summary(
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
        "passed": bool(pcc >= pcc_threshold and allclose),
        "pcc": float(pcc),
        "pcc_threshold": float(pcc_threshold),
        "allclose": allclose,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
    }


def _index_accuracy_summary(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    match_threshold: float,
) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(expected.shape):
        return {
            "passed": False,
            "reason": f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}",
        }
    mismatches = int((actual.to(torch.long) != expected.to(torch.long)).sum().item())
    total = int(expected.numel())
    match_fraction = float((total - mismatches) / total) if total else 1.0
    return {
        "passed": match_fraction >= match_threshold,
        "mismatch_count": mismatches,
        "total": total,
        "match_fraction": match_fraction,
        "match_threshold": float(match_threshold),
    }


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_flat = expected.reshape(-1).float()
    actual_flat = actual.reshape(-1).float()
    expected_centered = expected_flat - expected_flat.mean()
    actual_centered = actual_flat - actual_flat.mean()
    denominator = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if float(denominator.item()) == 0.0:
        return 1.0 if torch.allclose(expected_flat, actual_flat) else 0.0
    return float((expected_centered * actual_centered).sum().div(denominator).item())


def _validate_smoke_args(
    *,
    layer: int,
    expert: int,
    seq_len: int,
    max_tensors: int,
    max_bytes: int,
    norm_pcc: float,
    router_pcc: float,
    router_index_match: float,
    routed_pcc: float,
    shared_pcc: float,
    combined_pcc: float,
    residual_pcc: float,
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if expert < 0:
        raise ValueError(f"expert must be non-negative, got {expert}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in (
        ("norm_pcc", norm_pcc),
        ("router_pcc", router_pcc),
        ("router_index_match", router_index_match),
        ("routed_pcc", routed_pcc),
        ("shared_pcc", shared_pcc),
        ("combined_pcc", combined_pcc),
        ("residual_pcc", residual_pcc),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim != 4 or activation.shape[:2] != (1, 1):
        raise ValueError(f"activation must have shape [1, 1, tokens, hidden], got {tuple(activation.shape)}")
    if activation.shape[-1] != hidden_size:
        raise ValueError(f"activation hidden size must be {hidden_size}, got {activation.shape[-1]}")
    if activation.shape[-2] <= 0:
        raise ValueError("activation must contain at least one token")


if __name__ == "__main__":
    main()
