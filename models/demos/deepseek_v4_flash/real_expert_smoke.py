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
from models.demos.deepseek_v4_flash.cpu_reference import swiglu_expert
from models.demos.deepseek_v4_flash.expert_abi import PackedExpertWeight
from models.demos.deepseek_v4_flash.fp4 import EXPERT_FP4_BLOCK_SIZE, EXPERT_WEIGHT_ABI
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_EXPERT_MLP_EXPERT,
    DEFAULT_LAYER_EXPERT_MLP_LAYER,
    DEFAULT_MAX_BYTES,
    EXPERT_MLP_PROJECTIONS,
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_expert_mlp_keys,
)

REAL_EXPERT_SMOKE_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_EXPERT_MAX_TENSORS = 6


def run_real_expert_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_EXPERT_MLP_LAYER,
    expert: int = DEFAULT_LAYER_EXPERT_MLP_EXPERT,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_EXPERT_MAX_TENSORS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    output_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Load and execute one real DeepSeek V4 Flash routed expert MLP slice."""

    _validate_smoke_args(
        layer=layer,
        expert=expert,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        output_pcc=output_pcc,
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_real_expert_slice(
        snapshot_dir,
        layer=layer,
        expert=expert,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    weights = decode_real_expert_weights(tensors, config=config, layer=layer, expert=expert)

    activation = deterministic_expert_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    route_weight = deterministic_route_weight(seq_len=seq_len)
    reference_output = build_torch_expert_reference(
        weights,
        config=config,
        activation=activation,
        route_weight=route_weight,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        expert=expert,
        seq_len=seq_len,
        config=config,
        metadata=metadata,
        weights=weights,
        activation=activation,
        route_weight=route_weight,
        reference_output=reference_output,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    if seq_len % 32 != 0:
        raise ValueError(f"TTNN smoke seq_len must be a multiple of 32, got {seq_len}")

    ttnn_output = _run_ttnn_expert(
        weights,
        config=config,
        activation=activation,
        route_weight=route_weight,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "TtRoutedExpertMLP",
        "ttnn.linear(hidden,w1)",
        "ttnn.linear(hidden,w3)",
        "ttnn.mul(silu(gate),up)",
        "ttnn.mul(route_weight)",
        "ttnn.linear(hidden,w2)",
    ]
    result["ttnn"] = {"output": _tensor_summary(ttnn_output)}
    result["accuracy"] = {
        "expert_output": _accuracy_summary(
            reference_output,
            ttnn_output,
            pcc_threshold=output_pcc,
            rtol=rtol,
            atol=atol,
        )
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def load_real_expert_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    expert: int,
    max_tensors: int = DEFAULT_EXPERT_MAX_TENSORS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_expert_mlp_keys(index, layer=layer, expert=expert)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def decode_real_expert_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    validate_real_expert_slice(tensors, config=config, layer=layer, expert=expert)
    weights = {}
    for projection in EXPERT_MLP_PROJECTIONS:
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        packed = PackedExpertWeight(
            layer=layer,
            expert=expert,
            projection=projection,
            weight_packed=tensors[f"{prefix}.weight"],
            scale=tensors[f"{prefix}.scale"],
            abi=EXPERT_WEIGHT_ABI,
            block_size=EXPERT_FP4_BLOCK_SIZE,
        )
        weights[projection] = packed.dequantize(dtype=dtype)
    return weights


def validate_real_expert_slice(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    expert: int,
) -> None:
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    full_shapes = {
        "w1": (intermediate, hidden),
        "w2": (hidden, intermediate),
        "w3": (intermediate, hidden),
    }
    for projection, full_shape in full_shapes.items():
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.scale"
        if weight_key not in tensors:
            raise KeyError(f"Missing required real expert tensor {weight_key!r}")
        if scale_key not in tensors:
            raise KeyError(f"Missing required real expert tensor {scale_key!r}")

        rows, input_dim = full_shape
        if input_dim % 2 != 0:
            raise ValueError(f"{projection} input dim must be even for FP4 packing, got {input_dim}")
        if input_dim % EXPERT_FP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"{projection} input dim must be divisible by FP4 block size "
                f"{EXPERT_FP4_BLOCK_SIZE}, got {input_dim}"
            )

        expected_weight_shape = (rows, input_dim // 2)
        expected_scale_shape = (rows, input_dim // EXPERT_FP4_BLOCK_SIZE)
        weight = tensors[weight_key]
        scale = tensors[scale_key]
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(f"Expected {weight_key} shape {expected_weight_shape}, got {tuple(weight.shape)}")
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(f"Expected {scale_key} shape {expected_scale_shape}, got {tuple(scale.shape)}")
        if weight.dtype not in (torch.uint8, torch.int8):
            raise TypeError(f"Expected packed FP4 bytes for {weight_key}, got {weight.dtype}")


def deterministic_expert_activation(*, hidden_size: int, seq_len: int) -> torch.Tensor:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    values = torch.linspace(-0.015, 0.015, steps=seq_len * hidden_size, dtype=torch.float32)
    token_offsets = torch.linspace(-0.002, 0.002, steps=seq_len, dtype=torch.float32).repeat_interleave(hidden_size)
    activation = (values + token_offsets).reshape(1, 1, seq_len, hidden_size)
    return activation.to(torch.bfloat16)


def deterministic_route_weight(*, seq_len: int) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return torch.linspace(0.25, 1.0, steps=seq_len, dtype=torch.float32).reshape(1, seq_len, 1).to(torch.bfloat16)


def build_torch_expert_reference(
    weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    route_weight: torch.Tensor,
) -> torch.Tensor:
    _validate_activation(activation, hidden_size=config.hidden_size)
    if tuple(route_weight.shape) != (1, activation.shape[-2], 1):
        raise ValueError(
            f"route_weight must have shape {(1, activation.shape[-2], 1)}, got {tuple(route_weight.shape)}"
        )
    flat_output = swiglu_expert(
        activation[:, 0].reshape(-1, activation.shape[-1]),
        weights["w1"],
        weights["w2"],
        weights["w3"],
        route_weight=route_weight.reshape(-1, 1),
        swiglu_limit=config.swiglu_limit,
    )
    return flat_output.reshape(activation.shape[0], activation.shape[-2], activation.shape[-1]).unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real DeepSeek V4 Flash routed expert MLP TTNN smoke path.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_EXPERT_MLP_LAYER)
    parser.add_argument("--expert", type=int, default=DEFAULT_LAYER_EXPERT_MLP_EXPERT)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_EXPERT_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_expert_smoke(
        args.snapshot_dir,
        layer=args.layer,
        expert=args.expert,
        seq_len=args.seq_len,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        output_pcc=args.output_pcc,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_expert(
    weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    route_weight: torch.Tensor,
    device_id: int,
) -> torch.Tensor:
    import ttnn
    from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP

    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_input = ttnn.from_torch(
            activation,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_route_weight = ttnn.from_torch(
            route_weight.reshape(1, 1, route_weight.shape[1], 1)
            .expand(1, 1, route_weight.shape[1], weights["w1"].shape[0])
            .contiguous()
            .to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        module = TtRoutedExpertMLP(
            device=device,
            w1=weights["w1"],
            w2=weights["w2"],
            w3=weights["w3"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            swiglu_limit=config.swiglu_limit,
        )
        return ttnn.to_torch(module(tt_input, route_weight=tt_route_weight)).contiguous()
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
    weights: Mapping[str, torch.Tensor],
    activation: torch.Tensor,
    route_weight: torch.Tensor,
    reference_output: torch.Tensor,
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = int(sum(item.nbytes for item in metadata))
    return {
        "schema_version": REAL_EXPERT_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "expert": int(expert),
        "sequence_length": int(seq_len),
        "model": {
            "hidden_size": config.hidden_size,
            "moe_intermediate_size": config.moe_intermediate_size,
            "swiglu_limit": config.swiglu_limit,
            "expert_dtype": config.expert_dtype,
        },
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "payload_bytes": payload_bytes,
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": payload_bytes,
        },
        "expert_format": {
            "abi": EXPERT_WEIGHT_ABI,
            "block_size": EXPERT_FP4_BLOCK_SIZE,
            "packed_order": "low_nibble_first",
            "scale_axis": "input_blocks",
        },
        "decoded_tensors": {projection: _tensor_summary(weight) for projection, weight in weights.items()},
        "reference_ops": ["PackedExpertWeight.dequantize", "torch.swiglu_expert_reference"],
        "ttnn_ops": [],
        "inputs": {
            "activation": _tensor_summary(activation),
            "route_weight": _tensor_summary(route_weight),
        },
        "reference": {"output": _tensor_summary(reference_output)},
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _metadata_summary(item: TensorMetadata) -> dict[str, Any]:
    return {
        "canonical_key": item.canonical_key,
        "source_key": item.source_key,
        "shard": item.shard_name,
        "dtype": item.dtype,
        "shape": list(item.shape),
        "nbytes": item.nbytes,
    }


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
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
    output_pcc: float,
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
    if not 0.0 <= output_pcc <= 1.0:
        raise ValueError(f"output_pcc must be in [0, 1], got {output_pcc}")


def _validate_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim != 4 or activation.shape[:2] != (1, 1):
        raise ValueError(f"activation must have shape [1, 1, tokens, hidden], got {tuple(activation.shape)}")
    if activation.shape[-1] != hidden_size:
        raise ValueError(f"activation hidden size must be {hidden_size}, got {activation.shape[-1]}")


if __name__ == "__main__":
    main()
