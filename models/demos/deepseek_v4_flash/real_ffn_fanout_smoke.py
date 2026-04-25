# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import combine_routed_experts, rms_norm, v4_router
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_EXPERT_MLP_LAYER,
    DEFAULT_MAX_BYTES,
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_expert_mlp_keys,
    layer_shared_expert_mlp_keys,
)
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights, validate_real_expert_slice
from models.demos.deepseek_v4_flash.real_ffn_smoke import (
    _accuracy_summary,
    _index_accuracy_summary,
    _metadata_summary,
    _selection_summary,
    _shared_source_format_summary,
    _tensor_summary,
    _validate_router_norm_tensors,
)
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import (
    ROUTED_MOE_TTNN_TILE_MULTIPLE,
    build_torch_selected_expert_reference,
    deterministic_input_ids_for_expert,
    deterministic_routed_activation,
    selected_expert_route,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import (
    SHARED_EXPERT_FP8_BLOCK_SIZE,
    build_torch_shared_expert_reference,
    decode_real_shared_expert_weights,
    validate_real_shared_expert_slice,
)

REAL_FFN_FANOUT_SMOKE_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 1
DEFAULT_ANCHOR_EXPERT = 1
DEFAULT_FFN_FANOUT_MAX_TENSORS = 64
DEFAULT_FFN_FANOUT_MAX_BYTES = max(DEFAULT_MAX_BYTES, 128 * 1024 * 1024)
DEFAULT_ROUTER_EXPERT_CANDIDATE_MARGIN = 2


def run_real_ffn_fanout_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_EXPERT_MLP_LAYER,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    anchor_expert: int = DEFAULT_ANCHOR_EXPERT,
    max_tensors: int = DEFAULT_FFN_FANOUT_MAX_TENSORS,
    max_bytes: int = DEFAULT_FFN_FANOUT_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    norm_pcc: float = 0.999,
    router_pcc: float = 0.99,
    router_index_match: float = 1.0,
    routed_pcc: float = 0.99,
    shared_pcc: float = 0.99,
    combined_pcc: float = 0.99,
    residual_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Run all router-activated routed experts for a tiny real FFN decode slice."""

    _validate_smoke_args(
        layer=layer,
        seq_len=seq_len,
        anchor_expert=anchor_expert,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        pcc_thresholds={
            "norm_pcc": norm_pcc,
            "router_pcc": router_pcc,
            "router_index_match": router_index_match,
            "routed_pcc": routed_pcc,
            "shared_pcc": shared_pcc,
            "combined_pcc": combined_pcc,
            "residual_pcc": residual_pcc,
        },
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)

    selector_keys = layer_ffn_selector_keys(index, layer=layer)
    tensors, metadata = index.load_tensors(selector_keys, max_tensors=max_tensors, max_bytes=max_bytes)
    _validate_router_norm_tensors(tensors, config=config, layer=layer)

    activation, input_ids = _deterministic_decode_activation(
        tensors,
        config=config,
        layer=layer,
        seq_len=seq_len,
        anchor_expert=anchor_expert,
    )
    selector_reference = build_torch_ffn_fanout_selector_reference(
        tensors,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
    )
    activated_experts = _ordered_activated_expert_ids(selector_reference["router_indices"])
    fanout_keys = layer_ffn_fanout_keys(index, layer=layer, experts=activated_experts)
    tensors, metadata = _load_missing_tensors(
        index,
        tensors,
        metadata,
        fanout_keys,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    validate_real_ffn_fanout_slice(tensors, config=config, layer=layer, experts=activated_experts)
    routed_weights_by_expert = {
        expert: decode_real_expert_weights(tensors, config=config, layer=layer, expert=expert)
        for expert in activated_experts
    }
    shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=layer)

    reference = build_torch_ffn_fanout_reference(
        tensors,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        seq_len=seq_len,
        anchor_expert=anchor_expert,
        config=config,
        metadata=metadata,
        tensors=tensors,
        activation=activation,
        input_ids=input_ids,
        reference=reference,
        routed_weights_by_expert=routed_weights_by_expert,
        shared_weights=shared_weights,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    ttnn_outputs = _run_ttnn_ffn_fanout_slice(
        tensors,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "ttnn.rms_norm(ffn_norm)",
        "TtRouter(ttnn.linear+host_topk)",
        "host_gather_activated_expert_tokens",
        "sequential_TtRoutedExpertMLP_per_activated_expert",
        "host_scatter_add_activated_expert_outputs",
        "TtSharedExpertMLP",
        "host_add(full_routed_output,shared_expert_output)",
        "host_add(residual_input,combined_ffn_output)",
    ]
    result["ttnn"] = {
        "input_padding": ttnn_outputs["input_padding"],
        "rms_norm": _tensor_summary(ttnn_outputs["norm_output"]),
        "router_weights": _tensor_summary(ttnn_outputs["router_weights"]),
        "router_indices": _tensor_summary(ttnn_outputs["router_indices"]),
        "activated_experts": _fanout_summary(
            ttnn_outputs["router_weights"],
            ttnn_outputs["router_indices"],
            full_topk=config.num_experts_per_tok,
        ),
        "per_expert_routes": {
            str(expert): _selection_summary(route) for expert, route in ttnn_outputs["per_expert_routes"].items()
        },
        "per_expert_padding": {str(expert): padding for expert, padding in ttnn_outputs["per_expert_padding"].items()},
        "per_expert_selected_output": {
            str(expert): _tensor_summary(output)
            for expert, output in ttnn_outputs["per_expert_selected_output"].items()
        },
        "routed_output": _tensor_summary(ttnn_outputs["routed_output"]),
        "shared_output": _tensor_summary(ttnn_outputs["shared_output"]),
        "combined_output": _tensor_summary(ttnn_outputs["combined_output"]),
        "residual_output": _tensor_summary(ttnn_outputs["residual_output"]),
        "experts_executed": int(len(ttnn_outputs["per_expert_routes"])),
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


def layer_ffn_selector_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    prefix = f"layers.{layer}"
    keys = [
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.ffn.gate.weight",
    ]
    bias_key = f"{prefix}.ffn.gate.bias"
    tid2eid_key = f"{prefix}.ffn.gate.tid2eid"
    if index.has_tensor(bias_key):
        keys.append(bias_key)
    elif index.has_tensor(tid2eid_key):
        keys.append(tid2eid_key)
    else:
        raise KeyError(f"Layer {layer} has neither router bias nor tid2eid tensor in {index.snapshot_dir}")
    return keys


def layer_ffn_fanout_keys(index: RealCheckpointTensorIndex, *, layer: int, experts: Sequence[int]) -> list[str]:
    if not experts:
        raise ValueError("experts must contain at least one routed expert")
    keys = list(layer_ffn_selector_keys(index, layer=layer))
    for expert in experts:
        keys.extend(layer_expert_mlp_keys(index, layer=layer, expert=int(expert)))
    keys.extend(layer_shared_expert_mlp_keys(index, layer=layer))
    return _unique_keys(keys)


def validate_real_ffn_fanout_slice(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    experts: Sequence[int],
) -> None:
    _validate_router_norm_tensors(tensors, config=config, layer=layer)
    for expert in experts:
        validate_real_expert_slice(tensors, config=config, layer=layer, expert=int(expert))
    validate_real_shared_expert_slice(tensors, config=config, layer=layer)


def build_torch_ffn_fanout_selector_reference(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    input_ids: torch.Tensor | None,
    topk: int | None = None,
) -> dict[str, torch.Tensor | None]:
    _validate_activation(activation, hidden_size=config.hidden_size)
    route_topk = config.num_experts_per_tok if topk is None else int(topk)
    if route_topk <= 0 or route_topk > int(config.n_routed_experts):
        raise ValueError(f"topk must be in [1, {config.n_routed_experts}], got {route_topk}")
    prefix = f"layers.{layer}"
    norm_output = rms_norm(
        activation[:, 0],
        tensors[f"{prefix}.ffn_norm.weight"].to(torch.bfloat16),
        eps=config.rms_norm_eps,
    ).unsqueeze(1)
    router_weights, router_indices = v4_router(
        norm_output[:, 0],
        tensors[f"{prefix}.ffn.gate.weight"].to(torch.bfloat16),
        topk=route_topk,
        route_scale=config.routed_scaling_factor,
        scoring_func=config.scoring_func,
        bias=tensors.get(f"{prefix}.ffn.gate.bias"),
        input_ids=input_ids,
        tid2eid=tensors.get(f"{prefix}.ffn.gate.tid2eid"),
    )
    return {
        "norm_output": norm_output.to(torch.bfloat16),
        "router_weights": router_weights.contiguous(),
        "router_indices": router_indices.contiguous(),
        "input_ids": input_ids,
    }


def build_torch_ffn_fanout_candidate_experts(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    input_ids: torch.Tensor | None,
    candidate_margin: int = DEFAULT_ROUTER_EXPERT_CANDIDATE_MARGIN,
) -> list[int]:
    """Return experts to load, including near-boundary learned-router candidates."""

    if candidate_margin < 0:
        raise ValueError(f"candidate_margin must be non-negative, got {candidate_margin}")
    prefix = f"layers.{layer}"
    route_topk = int(config.num_experts_per_tok)
    if tensors.get(f"{prefix}.ffn.gate.tid2eid") is None:
        route_topk = min(int(config.n_routed_experts), route_topk + int(candidate_margin))
    selector = build_torch_ffn_fanout_selector_reference(
        tensors,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
        topk=route_topk,
    )
    return _ordered_activated_expert_ids(selector["router_indices"])


def build_torch_ffn_fanout_reference(
    tensors: Mapping[str, torch.Tensor],
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    selector = build_torch_ffn_fanout_selector_reference(
        tensors,
        config=config,
        layer=layer,
        activation=activation,
        input_ids=input_ids,
    )
    norm_output = selector["norm_output"]
    router_weights = selector["router_weights"]
    router_indices = selector["router_indices"]
    if not isinstance(norm_output, torch.Tensor) or not isinstance(router_weights, torch.Tensor):
        raise TypeError("selector reference did not produce tensor outputs")
    if not isinstance(router_indices, torch.Tensor):
        raise TypeError("selector reference did not produce router indices")

    activated_experts = _ordered_activated_expert_ids(router_indices)
    missing_experts = [expert for expert in activated_experts if expert not in routed_weights_by_expert]
    if missing_experts:
        raise KeyError(f"Missing routed weights for activated experts {missing_experts}")

    routed_output_float = torch.zeros_like(norm_output, dtype=torch.float32)
    per_expert: dict[int, dict[str, Any]] = {}
    for expert in activated_experts:
        route = selected_expert_route(router_weights, router_indices, expert=expert)
        expert_outputs = build_torch_selected_expert_reference(
            routed_weights_by_expert[expert],
            config=config,
            hidden_states=norm_output,
            selected_route=route,
        )
        routed_output_float += expert_outputs["expert_scattered_output"].float()
        per_expert[expert] = {
            "route": route,
            "selected_expert_input": expert_outputs["selected_expert_input"],
            "selected_expert_output": expert_outputs["selected_expert_output"],
            "expert_scattered_output": expert_outputs["expert_scattered_output"],
        }

    routed_output = routed_output_float.to(norm_output.dtype)
    shared_output = build_torch_shared_expert_reference(
        shared_weights,
        config=config,
        activation=norm_output,
    )
    combined_output = (routed_output.float() + shared_output.float()).to(norm_output.dtype)
    residual_output = (activation.float() + combined_output.float()).to(activation.dtype)
    manual_routed = combine_routed_experts(
        norm_output[:, 0],
        router_weights,
        router_indices,
        {
            expert: (
                weights["w1"],
                weights["w2"],
                weights["w3"],
            )
            for expert, weights in routed_weights_by_expert.items()
        },
        swiglu_limit=config.swiglu_limit,
    ).unsqueeze(1)
    return {
        "norm_output": norm_output,
        "router_weights": router_weights,
        "router_indices": router_indices,
        "input_ids": input_ids,
        "activated_experts": activated_experts,
        "per_expert": per_expert,
        "routed_output": routed_output,
        "manual_routed_output": manual_routed,
        "shared_output": shared_output,
        "combined_output": combined_output,
        "residual_output": residual_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a DeepSeek V4 Flash one-token FFN slice with full routed expert fanout/combine."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_EXPERT_MLP_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--anchor-expert", type=int, default=DEFAULT_ANCHOR_EXPERT)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_FFN_FANOUT_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_FFN_FANOUT_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--norm-pcc", type=float, default=0.999)
    parser.add_argument("--router-pcc", type=float, default=0.99)
    parser.add_argument("--router-index-match", type=float, default=1.0)
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

    result = run_real_ffn_fanout_smoke(
        args.snapshot_dir,
        layer=args.layer,
        seq_len=args.seq_len,
        anchor_expert=args.anchor_expert,
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


def _run_ttnn_ffn_fanout_slice(
    tensors: Mapping[str, torch.Tensor],
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    input_ids: torch.Tensor | None,
    device_id: int,
) -> dict[str, Any]:
    import ttnn
    from models.demos.deepseek_v4_flash.real_routed_moe_smoke import _pad_selected_expert_inputs
    from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP
    from models.demos.deepseek_v4_flash.ttnn_router import TtRouter
    from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP

    prefix = f"layers.{layer}"
    real_tokens = int(activation.shape[-2])
    padded_activation, padding = _pad_token_dim(activation, tile_multiple=ROUTED_MOE_TTNN_TILE_MULTIPLE)
    padded_input_ids = _pad_input_ids(input_ids, padded_tokens=padded_activation.shape[-2])
    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_input = ttnn.from_torch(
            padded_activation,
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
        router = TtRouter(
            device=device,
            gate_weight=tensors[f"{prefix}.ffn.gate.weight"],
            bias=tensors.get(f"{prefix}.ffn.gate.bias"),
            tid2eid=tensors.get(f"{prefix}.ffn.gate.tid2eid"),
            topk=config.num_experts_per_tok,
            route_scale=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        router_weights, router_indices = router(tt_norm, input_ids=padded_input_ids)
        router_weights = router_weights[:, :real_tokens, :].contiguous()
        router_indices = router_indices[:, :real_tokens, :].contiguous()
        activated_experts = _ordered_activated_expert_ids(router_indices)
        missing_experts = [expert for expert in activated_experts if expert not in routed_weights_by_expert]
        if missing_experts:
            raise KeyError(f"TTNN router selected unloaded experts {missing_experts}")

        norm_output = ttnn.to_torch(tt_norm).contiguous()[:, :, :real_tokens, :]
        routed_output_float = torch.zeros_like(norm_output, dtype=torch.float32)
        per_expert_routes = {}
        per_expert_padding = {}
        per_expert_selected_output = {}
        for expert in activated_experts:
            expert_weights = routed_weights_by_expert[expert]
            selected_route = selected_expert_route(router_weights, router_indices, expert=expert)
            selected_input = norm_output[:, :, selected_route.token_indices, :].contiguous()
            padded_input, padded_route_weight, expert_padding = _pad_selected_expert_inputs(
                selected_input,
                selected_route.route_weight,
                intermediate_size=expert_weights["w1"].shape[0],
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
                w1=expert_weights["w1"],
                w2=expert_weights["w2"],
                w3=expert_weights["w3"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                swiglu_limit=config.swiglu_limit,
            )
            tt_padded_output = routed_module(tt_selected_input, route_weight=tt_route_weight)
            padded_output = ttnn.to_torch(tt_padded_output).contiguous()
            selected_output = padded_output[:, :, : selected_route.selected_token_count, :].contiguous()
            scattered = torch.zeros_like(norm_output, dtype=torch.float32)
            scattered[:, :, selected_route.token_indices, :] = selected_output.float()
            routed_output_float += scattered
            per_expert_routes[expert] = selected_route
            per_expert_padding[expert] = expert_padding
            per_expert_selected_output[expert] = selected_output
            _safe_deallocate(
                ttnn,
                tt_selected_input,
                tt_route_weight,
                tt_padded_output,
                routed_module.w1,
                routed_module.w2,
                routed_module.w3,
            )

        shared_module = TtSharedExpertMLP(
            device=device,
            w1=shared_weights["w1"],
            w2=shared_weights["w2"],
            w3=shared_weights["w3"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            swiglu_limit=config.swiglu_limit,
        )
        tt_shared_output = shared_module(tt_norm)
        shared_output = ttnn.to_torch(tt_shared_output).contiguous()[:, :, :real_tokens, :]
        routed_output = routed_output_float.to(norm_output.dtype)
        combined_output = (routed_output.float() + shared_output.float()).to(norm_output.dtype)
        residual_output = (activation.float() + combined_output.float()).to(activation.dtype)
        return {
            "input_padding": padding,
            "norm_output": norm_output.contiguous(),
            "router_weights": router_weights,
            "router_indices": router_indices,
            "per_expert_routes": per_expert_routes,
            "per_expert_padding": per_expert_padding,
            "per_expert_selected_output": per_expert_selected_output,
            "routed_output": routed_output.contiguous(),
            "shared_output": shared_output.contiguous(),
            "combined_output": combined_output.contiguous(),
            "residual_output": residual_output.contiguous(),
        }
    finally:
        ttnn.close_device(device)


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    seq_len: int,
    anchor_expert: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    tensors: Mapping[str, torch.Tensor],
    activation: torch.Tensor,
    input_ids: torch.Tensor | None,
    reference: Mapping[str, Any],
    routed_weights_by_expert: Mapping[int, Mapping[str, torch.Tensor]],
    shared_weights: Mapping[str, torch.Tensor],
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = _payload_byte_split(metadata)
    activated_experts = reference["activated_experts"]
    return {
        "schema_version": REAL_FFN_FANOUT_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "sequence_length": int(seq_len),
        "decode_tokens": int(seq_len),
        "activation_source": "deterministic_one_token_ffn_decode_activation",
        "anchor_expert": int(anchor_expert),
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
        "residual_semantics": (
            "residual_output = activation + (sum(route_weight_i * routed_expert_i(norm_output)) "
            "+ shared_expert(norm_output))"
        ),
        "fanout_scope": {
            "full_expert_fanout": "enabled for every router top-k expert selected by this token",
            "activated_expert_count": len(activated_experts),
            "activated_expert_ids": [int(expert) for expert in activated_experts],
            "topk": int(config.num_experts_per_tok),
            "topk_prefix_limit": None,
        },
        "performance_boundaries": [
            {
                "name": "sequential_expert_execution",
                "description": (
                    "activated routed experts run one after another on TTNN for this correctness slice; "
                    "parallel expert dispatch/combine remains the next performance target"
                ),
            }
        ],
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "payload_bytes": payload_bytes,
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": payload_bytes["total"],
        },
        "expert_format": {
            "routed": {
                "format": "PACKED_FP4",
                "activated_experts": {
                    str(expert): {projection: _tensor_summary(weight) for projection, weight in weights.items()}
                    for expert, weights in routed_weights_by_expert.items()
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
                "description": "only activated routed expert FP4 weights are decoded on host to BF16",
            },
            {
                "name": "shared_fp8_decode_to_bf16",
                "location": "before TtSharedExpertMLP",
                "description": "shared expert FP8 weights and scales are decoded on host to BF16",
            },
            {
                "name": "router_topk",
                "location": "TtRouter",
                "description": "router scores leave device for host DeepSeek top-k/hash selection",
            },
            {
                "name": "normalized_hidden_readback",
                "location": "between router and routed experts",
                "description": "normalized hidden states are copied to host to gather per-expert token slices",
            },
            {
                "name": "activated_expert_gather",
                "location": "between router and routed experts",
                "description": "token activations and route weights are gathered on host for each activated expert",
            },
            {
                "name": "activated_expert_tile_padding",
                "location": "before each TtRoutedExpertMLP",
                "description": "each selected-token batch is padded to a TTNN tile multiple with zero route weight",
            },
            {
                "name": "activated_expert_scatter_add",
                "location": "after each TtRoutedExpertMLP",
                "description": "all activated expert contributions are scattered and accumulated on host",
            },
            {
                "name": "shared_expert_readback",
                "location": "after TtSharedExpertMLP",
                "description": "shared expert output is copied to host for this smoke-test combine and comparison",
            },
            {
                "name": "ffn_host_combine",
                "location": "after routed and shared experts",
                "description": "full routed fanout output and shared expert output are added on host",
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
            "host_gather_activated_expert_tokens",
            "torch.routed_swiglu_expert_reference_per_activated_expert",
            "host_scatter_add_activated_expert_outputs",
            "torch.shared_swiglu_expert_reference",
            "host_add(full_routed_output,shared_expert_output)",
            "host_add(residual_input,combined_ffn_output)",
        ],
        "ttnn_ops": [],
        "inputs": {
            "activation": _tensor_summary(activation),
            "input_ids": _tensor_summary(input_ids),
        },
        "reference": {
            "rms_norm": _tensor_summary(reference["norm_output"]),
            "router_weights": _tensor_summary(reference["router_weights"]),
            "router_indices": _tensor_summary(reference["router_indices"]),
            "activated_experts": _fanout_summary(
                reference["router_weights"],
                reference["router_indices"],
                full_topk=config.num_experts_per_tok,
            ),
            "per_expert_routes": {
                str(expert): _selection_summary(values["route"]) for expert, values in reference["per_expert"].items()
            },
            "per_expert_selected_output": {
                str(expert): _tensor_summary(values["selected_expert_output"])
                for expert, values in reference["per_expert"].items()
            },
            "routed_output": _tensor_summary(reference["routed_output"]),
            "manual_routed_output": _tensor_summary(reference["manual_routed_output"]),
            "shared_output": _tensor_summary(reference["shared_output"]),
            "combined_output": _tensor_summary(reference["combined_output"]),
            "residual_output": _tensor_summary(reference["residual_output"]),
        },
        "router_match_stats": {},
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _deterministic_decode_activation(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    seq_len: int,
    anchor_expert: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prefix = f"layers.{layer}"
    activation = deterministic_routed_activation(
        hidden_size=config.hidden_size,
        seq_len=seq_len,
        gate_weight=tensors[f"{prefix}.ffn.gate.weight"],
        expert=anchor_expert,
    )
    tid2eid = tensors.get(f"{prefix}.ffn.gate.tid2eid")
    input_ids = (
        deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=anchor_expert, seq_len=seq_len)
        if tid2eid is not None
        else None
    )
    return activation, input_ids


def _ordered_activated_expert_ids(route_indices: torch.Tensor) -> list[int]:
    if route_indices.ndim != 3 or route_indices.shape[0] != 1:
        raise ValueError(f"route_indices must have shape [1, tokens, topk], got {tuple(route_indices.shape)}")
    expert_ids: list[int] = []
    for value in route_indices.reshape(-1).to(torch.long).tolist():
        expert = int(value)
        if expert not in expert_ids:
            expert_ids.append(expert)
    if not expert_ids:
        raise ValueError("router did not activate any experts")
    return expert_ids


def _fanout_summary(route_weights: torch.Tensor, route_indices: torch.Tensor, *, full_topk: int) -> dict[str, Any]:
    if route_weights.shape != route_indices.shape:
        raise ValueError(
            f"route_weights and route_indices must have the same shape, got "
            f"{tuple(route_weights.shape)} and {tuple(route_indices.shape)}"
        )
    activated_experts = _ordered_activated_expert_ids(route_indices)
    return {
        "activated_expert_ids": [int(expert) for expert in activated_experts],
        "activated_expert_count": len(activated_experts),
        "full_topk": int(full_topk),
        "executed_topk": int(route_indices.shape[-1]),
        "topk_is_full": int(route_indices.shape[-1]) == int(full_topk),
        "topk_expert_ids_by_token": [[int(value) for value in row] for row in route_indices[0].to(torch.long).tolist()],
        "router_weights_by_token": [[float(value) for value in row] for row in route_weights[0].float().tolist()],
    }


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
    loaded_tensors, loaded_metadata = index.load_tensors(
        missing_keys,
        max_tensors=max_tensors - len(metadata),
        max_bytes=max_bytes - used_bytes,
    )
    tensors.update(loaded_tensors)
    return tensors, [*metadata, *loaded_metadata]


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, Any]:
    split: dict[str, Any] = {
        "norm": 0,
        "router": 0,
        "routed_experts": 0,
        "routed_experts_by_id": {},
        "shared_expert": 0,
    }
    by_id: dict[str, int] = {}
    for item in metadata:
        if ".ffn_norm." in item.canonical_key:
            split["norm"] += item.nbytes
        elif ".ffn.gate." in item.canonical_key:
            split["router"] += item.nbytes
        elif ".ffn.experts." in item.canonical_key:
            expert = item.canonical_key.split(".ffn.experts.", 1)[1].split(".", 1)[0]
            by_id[expert] = by_id.get(expert, 0) + item.nbytes
            split["routed_experts"] += item.nbytes
        elif ".ffn.shared_experts." in item.canonical_key:
            split["shared_expert"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in FFN fanout slice: {item.canonical_key}")
    split["routed_experts_by_id"] = dict(sorted(by_id.items(), key=lambda item: int(item[0])))
    split["total"] = int(split["norm"] + split["router"] + split["routed_experts"] + split["shared_expert"])
    return split


def _pad_token_dim(tensor: torch.Tensor, *, tile_multiple: int) -> tuple[torch.Tensor, dict[str, int]]:
    tokens = int(tensor.shape[-2])
    padded_tokens = int(math.ceil(tokens / tile_multiple) * tile_multiple)
    pad_tokens = padded_tokens - tokens
    if pad_tokens:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_tokens))
    return (
        tensor.contiguous().to(torch.bfloat16),
        {
            "tile_multiple": int(tile_multiple),
            "tokens": tokens,
            "padded_tokens": padded_tokens,
            "pad_tokens": pad_tokens,
        },
    )


def _pad_input_ids(input_ids: torch.Tensor | None, *, padded_tokens: int) -> torch.Tensor | None:
    if input_ids is None:
        return None
    real_tokens = int(input_ids.shape[-1])
    if real_tokens == padded_tokens:
        return input_ids.contiguous()
    if real_tokens <= 0:
        raise ValueError("input_ids must contain at least one token")
    pad_tokens = padded_tokens - real_tokens
    if pad_tokens < 0:
        raise ValueError(f"input_ids length {real_tokens} exceeds padded token count {padded_tokens}")
    pad_value = int(input_ids[0, -1].item())
    padding = input_ids.new_full((input_ids.shape[0], pad_tokens), pad_value)
    return torch.cat([input_ids, padding], dim=-1).contiguous()


def _safe_deallocate(ttnn_module, *tensors) -> None:
    deallocate = getattr(ttnn_module, "deallocate", None)
    if deallocate is None:
        return
    for tensor in tensors:
        try:
            deallocate(tensor)
        except Exception:
            pass


def _unique_keys(keys: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for key in keys:
        if key not in unique:
            unique.append(key)
    return unique


def _validate_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim != 4 or activation.shape[:2] != (1, 1):
        raise ValueError(f"activation must have shape [1, 1, tokens, hidden], got {tuple(activation.shape)}")
    if activation.shape[-1] != hidden_size:
        raise ValueError(f"activation hidden size must be {hidden_size}, got {activation.shape[-1]}")
    if activation.shape[-2] <= 0:
        raise ValueError("activation must contain at least one token")


def _validate_smoke_args(
    *,
    layer: int,
    seq_len: int,
    anchor_expert: int,
    max_tensors: int,
    max_bytes: int,
    pcc_thresholds: Mapping[str, float],
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if anchor_expert < 0:
        raise ValueError(f"anchor_expert must be non-negative, got {anchor_expert}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


if __name__ == "__main__":
    main()
