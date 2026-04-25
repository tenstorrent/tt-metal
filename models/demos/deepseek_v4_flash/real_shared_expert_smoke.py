# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
from models.demos.deepseek_v4_flash.cpu_reference import swiglu_expert
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_SHARED_EXPERT_MLP_LAYER,
    SHARED_EXPERT_MLP_PROJECTIONS,
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_shared_expert_mlp_keys,
)

REAL_SHARED_EXPERT_SMOKE_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_SHARED_EXPERT_MAX_TENSORS = 6
DEFAULT_SHARED_EXPERT_MAX_BYTES = 32 * 1024 * 1024
SHARED_EXPERT_FP8_BLOCK_SIZE = (128, 128)
SHARED_EXPERT_TTNN_TILE_MULTIPLE = 32
_DIRECT_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def run_real_shared_expert_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_SHARED_EXPERT_MLP_LAYER,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_SHARED_EXPERT_MAX_TENSORS,
    max_bytes: int = DEFAULT_SHARED_EXPERT_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    output_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Load and execute one real DeepSeek V4 Flash shared expert MLP slice."""

    _validate_smoke_args(
        layer=layer,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        output_pcc=output_pcc,
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_real_shared_expert_slice(
        snapshot_dir,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    weights = decode_real_shared_expert_weights(tensors, config=config, layer=layer)

    activation = deterministic_shared_expert_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    reference_output = build_torch_shared_expert_reference(
        weights,
        config=config,
        activation=activation,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        seq_len=seq_len,
        config=config,
        metadata=metadata,
        tensors=tensors,
        weights=weights,
        activation=activation,
        reference_output=reference_output,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    if seq_len % SHARED_EXPERT_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN smoke seq_len must be a multiple of {SHARED_EXPERT_TTNN_TILE_MULTIPLE}, got {seq_len}")

    ttnn_output = _run_ttnn_shared_expert(
        weights,
        config=config,
        activation=activation,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "TtSharedExpertMLP",
        "ttnn.linear(hidden,w1)",
        "ttnn.linear(hidden,w3)",
        "ttnn.mul(silu(gate),up)",
        "ttnn.linear(hidden,w2)",
    ]
    result["ttnn"] = {"output": _tensor_summary(ttnn_output)}
    result["accuracy"] = {
        "shared_expert_output": _accuracy_summary(
            reference_output,
            ttnn_output,
            pcc_threshold=output_pcc,
            rtol=rtol,
            atol=atol,
        )
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def load_real_shared_expert_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    max_tensors: int = DEFAULT_SHARED_EXPERT_MAX_TENSORS,
    max_bytes: int = DEFAULT_SHARED_EXPERT_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_shared_expert_mlp_keys(index, layer=layer)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def decode_real_shared_expert_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    validate_real_shared_expert_slice(tensors, config=config, layer=layer)
    weights = {}
    for projection in SHARED_EXPERT_MLP_PROJECTIONS:
        prefix = f"layers.{layer}.ffn.shared_experts.{projection}"
        weight = tensors[f"{prefix}.weight"]
        scale = tensors[f"{prefix}.scale"]
        if _is_fp8_shared_expert_tensor(weight, scale):
            weights[projection] = decode_fp8_block_scaled_weight(
                weight,
                scale,
                block_size=SHARED_EXPERT_FP8_BLOCK_SIZE,
                dtype=dtype,
            )
        elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
            weights[projection] = weight.contiguous().to(dtype)
        else:
            raise TypeError(
                f"Unsupported shared expert format for {prefix}: "
                f"weight dtype {weight.dtype}, scale dtype {scale.dtype}"
            )
    return weights


def decode_fp8_block_scaled_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: tuple[int, int] = SHARED_EXPERT_FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"FP8 shared expert weight must be rank 2, got {tuple(weight.shape)}")
    if scale.ndim != 2:
        raise ValueError(f"FP8 shared expert scale must be rank 2, got {tuple(scale.shape)}")
    block_rows, block_cols = (int(block_size[0]), int(block_size[1]))
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(f"block_size entries must be positive, got {block_size}")

    rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
    expected_scale_shape = (math.ceil(rows / block_rows), math.ceil(cols / block_cols))
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"Expected FP8 scale shape {expected_scale_shape}, got {tuple(scale.shape)}")

    scale_values = scale.float().repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)
    scale_values = scale_values[:rows, :cols]
    return (weight.float() * scale_values).to(dtype).contiguous()


def validate_real_shared_expert_slice(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> None:
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size) * int(config.n_shared_experts)
    full_shapes = {
        "w1": (intermediate, hidden),
        "w2": (hidden, intermediate),
        "w3": (intermediate, hidden),
    }
    for projection, full_shape in full_shapes.items():
        prefix = f"layers.{layer}.ffn.shared_experts.{projection}"
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.scale"
        if weight_key not in tensors:
            raise KeyError(f"Missing required real shared expert tensor {weight_key!r}")
        if scale_key not in tensors:
            raise KeyError(f"Missing required real shared expert tensor {scale_key!r}")

        weight = tensors[weight_key]
        scale = tensors[scale_key]
        if tuple(weight.shape) != full_shape:
            raise ValueError(f"Expected {weight_key} shape {full_shape}, got {tuple(weight.shape)}")
        if _is_fp8_shared_expert_tensor(weight, scale):
            expected_scale_shape = (
                math.ceil(full_shape[0] / SHARED_EXPERT_FP8_BLOCK_SIZE[0]),
                math.ceil(full_shape[1] / SHARED_EXPERT_FP8_BLOCK_SIZE[1]),
            )
            if tuple(scale.shape) != expected_scale_shape:
                raise ValueError(f"Expected {scale_key} shape {expected_scale_shape}, got {tuple(scale.shape)}")
        elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
            if scale.ndim != 2:
                raise ValueError(f"Expected {scale_key} to be rank 2, got {tuple(scale.shape)}")
        else:
            raise TypeError(
                f"Unsupported shared expert format for {prefix}: "
                f"weight dtype {weight.dtype}, scale dtype {scale.dtype}"
            )


def deterministic_shared_expert_activation(*, hidden_size: int, seq_len: int) -> torch.Tensor:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    values = torch.linspace(-1.0, 1.0, steps=seq_len * hidden_size, dtype=torch.float32)
    activation = values.reshape(seq_len, hidden_size)
    activation = activation - activation.mean(dim=-1, keepdim=True)
    activation = activation * torch.rsqrt(activation.square().mean(dim=-1, keepdim=True) + 1e-6)
    token_scales = torch.linspace(0.9, 1.1, steps=seq_len, dtype=torch.float32).reshape(seq_len, 1)
    activation = activation * token_scales
    return activation.reshape(1, 1, seq_len, hidden_size).to(torch.bfloat16)


def build_torch_shared_expert_reference(
    weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
) -> torch.Tensor:
    _validate_activation(activation, hidden_size=config.hidden_size)
    flat_output = swiglu_expert(
        activation[:, 0].reshape(-1, activation.shape[-1]),
        weights["w1"],
        weights["w2"],
        weights["w3"],
        swiglu_limit=config.swiglu_limit,
    )
    return flat_output.reshape(activation.shape[0], activation.shape[-2], activation.shape[-1]).unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real DeepSeek V4 Flash shared expert MLP TTNN smoke path.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_SHARED_EXPERT_MLP_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_SHARED_EXPERT_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_SHARED_EXPERT_MAX_BYTES)
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

    result = run_real_shared_expert_smoke(
        args.snapshot_dir,
        layer=args.layer,
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


def _run_ttnn_shared_expert(
    weights: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    device_id: int,
) -> torch.Tensor:
    import ttnn
    from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP

    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_input = ttnn.from_torch(
            activation,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        module = TtSharedExpertMLP(
            device=device,
            w1=weights["w1"],
            w2=weights["w2"],
            w3=weights["w3"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            swiglu_limit=config.swiglu_limit,
        )
        return ttnn.to_torch(module(tt_input)).contiguous()
    finally:
        ttnn.close_device(device)


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    seq_len: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    tensors: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    activation: torch.Tensor,
    reference_output: torch.Tensor,
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = _payload_byte_split(metadata)
    return {
        "schema_version": REAL_SHARED_EXPERT_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "sequence_length": int(seq_len),
        "model": {
            "hidden_size": config.hidden_size,
            "moe_intermediate_size": config.moe_intermediate_size,
            "n_shared_experts": config.n_shared_experts,
            "shared_intermediate_size": int(config.moe_intermediate_size) * int(config.n_shared_experts),
            "swiglu_limit": config.swiglu_limit,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "payload_bytes": payload_bytes,
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": payload_bytes["total"],
        },
        "shared_expert_format": {
            "weight_block_size": list(SHARED_EXPERT_FP8_BLOCK_SIZE),
            "source_formats": _source_format_summary(tensors, layer=layer),
        },
        "decoded_tensors": {projection: _tensor_summary(weight) for projection, weight in weights.items()},
        "host_boundaries": [
            {
                "name": "fp8_decode_to_bf16",
                "location": "before TtSharedExpertMLP",
                "description": "FP8 E4M3 shared expert weights and UE8M0 block scales are decoded on host to BF16",
            },
            {
                "name": "ttnn_output_readback",
                "location": "after TtSharedExpertMLP",
                "description": "TTNN output is copied to host only for this smoke-test accuracy comparison",
            },
        ],
        "reference_ops": ["decode_shared_expert_weights", "torch.swiglu_expert_reference"],
        "ttnn_ops": [],
        "inputs": {"activation": _tensor_summary(activation)},
        "reference": {"output": _tensor_summary(reference_output)},
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _is_fp8_shared_expert_tensor(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    return weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {"weights": 0, "scales": 0}
    for item in metadata:
        if ".ffn.shared_experts." not in item.canonical_key:
            raise ValueError(f"Unexpected tensor in shared expert slice: {item.canonical_key}")
        if item.canonical_key.endswith(".weight"):
            split["weights"] += item.nbytes
        elif item.canonical_key.endswith(".scale"):
            split["scales"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in shared expert slice: {item.canonical_key}")
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


def _source_format_summary(tensors: Mapping[str, torch.Tensor], *, layer: int) -> dict[str, Any]:
    formats = {}
    for projection in SHARED_EXPERT_MLP_PROJECTIONS:
        prefix = f"layers.{layer}.ffn.shared_experts.{projection}"
        weight = tensors[f"{prefix}.weight"]
        scale = tensors[f"{prefix}.scale"]
        if _is_fp8_shared_expert_tensor(weight, scale):
            source_format = "FP8_E4M3_WEIGHT_UE8M0_128x128_SCALE"
        elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
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
    seq_len: int,
    max_tensors: int,
    max_bytes: int,
    output_pcc: float,
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
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
    if activation.shape[-2] <= 0:
        raise ValueError("activation must contain at least one token")


if __name__ == "__main__":
    main()
