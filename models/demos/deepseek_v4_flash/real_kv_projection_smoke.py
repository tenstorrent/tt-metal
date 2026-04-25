# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, TensorMetadata
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_fp8_block_scaled_weight

REAL_KV_PROJECTION_SMOKE_SCHEMA_VERSION = 1
DEFAULT_LAYER_KV_PROJECTION_LAYER = 3
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_KV_PROJECTION_MAX_TENSORS = 4
DEFAULT_KV_PROJECTION_MAX_BYTES = 8 * 1024 * 1024
KV_FP8_BLOCK_SIZE = (128, 128)
KV_TTNN_TILE_MULTIPLE = 32
_DIRECT_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@dataclass(frozen=True)
class KvProjectionWeights:
    wkv: torch.Tensor
    kv_norm: torch.Tensor


def run_real_kv_projection_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_KV_PROJECTION_LAYER,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_KV_PROJECTION_MAX_TENSORS,
    max_bytes: int = DEFAULT_KV_PROJECTION_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    attn_norm_pcc: float = 0.999,
    kv_linear_pcc: float = 0.99,
    kv_output_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Load and execute one real DeepSeek V4 Flash attention K/V projection slice."""

    _validate_smoke_args(
        layer=layer,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        attn_norm_pcc=attn_norm_pcc,
        kv_linear_pcc=kv_linear_pcc,
        kv_output_pcc=kv_output_pcc,
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_real_kv_projection_slice(
        snapshot_dir,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    weights = decode_real_kv_projection_weights(tensors, config=config, layer=layer)

    activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    reference = build_torch_kv_projection_reference(
        tensors,
        weights,
        config=config,
        layer=layer,
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
        reference=reference,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    if seq_len % KV_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN smoke seq_len must be a multiple of {KV_TTNN_TILE_MULTIPLE}, got {seq_len}")

    ttnn_outputs = _run_ttnn_kv_projection(
        tensors,
        weights,
        config=config,
        layer=layer,
        activation=activation,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "ttnn.rms_norm(attn_norm)",
        "ttnn.linear(wkv)",
        "ttnn.rms_norm(kv_norm)",
    ]
    result["ttnn"] = {
        "attn_norm_output": _tensor_summary(ttnn_outputs["attn_norm_output"]),
        "kv_linear": _tensor_summary(ttnn_outputs["kv_linear"]),
        "kv_output": _tensor_summary(ttnn_outputs["kv_output"]),
    }
    result["accuracy"] = {
        "attn_norm_output": _accuracy_summary(
            reference["attn_norm_output"],
            ttnn_outputs["attn_norm_output"],
            pcc_threshold=attn_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "kv_linear": _accuracy_summary(
            reference["kv_linear"],
            ttnn_outputs["kv_linear"],
            pcc_threshold=kv_linear_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "kv_output": _accuracy_summary(
            reference["kv_output"],
            ttnn_outputs["kv_output"],
            pcc_threshold=kv_output_pcc,
            rtol=rtol,
            atol=atol,
        ),
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_kv_projection_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")

    prefix = f"layers.{layer}.attn"
    required_keys = [
        f"layers.{layer}.attn_norm.weight",
        f"{prefix}.kv_norm.weight",
        f"{prefix}.wkv.weight",
    ]
    for key in required_keys:
        index.location(key)

    keys = list(required_keys)
    wkv_metadata = index.metadata_for_keys([f"{prefix}.wkv.weight"])[0]
    if _metadata_dtype_is_fp8(wkv_metadata.dtype):
        scale_key = f"{prefix}.wkv.scale"
        index.location(scale_key)
        keys.append(scale_key)
    return keys


def load_real_kv_projection_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    max_tensors: int = DEFAULT_KV_PROJECTION_MAX_TENSORS,
    max_bytes: int = DEFAULT_KV_PROJECTION_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_kv_projection_keys(index, layer=layer)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def decode_real_kv_projection_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    dtype: torch.dtype = torch.bfloat16,
) -> KvProjectionWeights:
    validate_real_kv_projection_slice(tensors, config=config, layer=layer)
    prefix = f"layers.{layer}.attn"
    weight = tensors[f"{prefix}.wkv.weight"]
    scale = tensors.get(f"{prefix}.wkv.scale")
    if _is_fp8_kv_tensor(weight, scale):
        decoded_wkv = decode_fp8_block_scaled_weight(
            weight,
            scale,
            block_size=KV_FP8_BLOCK_SIZE,
            dtype=dtype,
        )
    elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
        decoded_wkv = weight.contiguous().to(dtype)
    else:
        scale_dtype = None if scale is None else scale.dtype
        raise TypeError(
            f"Unsupported K/V projection format for {prefix}.wkv: "
            f"weight dtype {weight.dtype}, scale dtype {scale_dtype}"
        )
    return KvProjectionWeights(
        wkv=decoded_wkv,
        kv_norm=tensors[f"{prefix}.kv_norm.weight"].contiguous().to(dtype),
    )


def validate_real_kv_projection_slice(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> None:
    hidden = int(config.hidden_size)
    kv_output_dim = _kv_output_dim(config)
    prefix = f"layers.{layer}.attn"
    expected_shapes = {
        f"layers.{layer}.attn_norm.weight": (hidden,),
        f"{prefix}.kv_norm.weight": (kv_output_dim,),
        f"{prefix}.wkv.weight": (kv_output_dim, hidden),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in tensors:
            raise KeyError(f"Missing required real K/V projection tensor {key!r}")
        if tuple(tensors[key].shape) != expected_shape:
            raise ValueError(f"Expected {key} shape {expected_shape}, got {tuple(tensors[key].shape)}")

    weight_key = f"{prefix}.wkv.weight"
    scale_key = f"{prefix}.wkv.scale"
    weight = tensors[weight_key]
    scale = tensors.get(scale_key)
    if _is_fp8_kv_tensor(weight, scale):
        expected_scale_shape = (
            math.ceil(weight.shape[0] / KV_FP8_BLOCK_SIZE[0]),
            math.ceil(weight.shape[1] / KV_FP8_BLOCK_SIZE[1]),
        )
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(f"Expected {scale_key} shape {expected_scale_shape}, got {tuple(scale.shape)}")
    elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
        if scale is not None and scale.ndim != 2:
            raise ValueError(f"Expected {scale_key} to be rank 2, got {tuple(scale.shape)}")
    else:
        scale_dtype = None if scale is None else scale.dtype
        raise TypeError(
            f"Unsupported K/V projection format for {prefix}.wkv: "
            f"weight dtype {weight.dtype}, scale dtype {scale_dtype}"
        )


def build_torch_kv_projection_reference(
    tensors: Mapping[str, torch.Tensor],
    weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_activation(activation, hidden_size=config.hidden_size)
    prefix = f"layers.{layer}.attn"
    attn_norm_weight = tensors[f"layers.{layer}.attn_norm.weight"].to(torch.bfloat16)
    attn_norm_output = rms_norm(activation[:, 0], attn_norm_weight, eps=config.rms_norm_eps).unsqueeze(1)
    kv_linear = F.linear(attn_norm_output[:, 0].float(), weights.wkv.float()).to(torch.bfloat16)
    kv_output = rms_norm(kv_linear, weights.kv_norm, config.rms_norm_eps).unsqueeze(1)
    return {
        "attn_norm_output": attn_norm_output,
        "kv_linear": kv_linear.unsqueeze(1),
        "kv_output": kv_output,
        "cache_projection": kv_output[:, 0].contiguous(),
        "attn_norm_weight": attn_norm_weight,
        "kv_norm_weight": tensors[f"{prefix}.kv_norm.weight"].to(torch.bfloat16),
    }


def deterministic_attention_activation(*, hidden_size: int, seq_len: int) -> torch.Tensor:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    values = torch.linspace(-0.75, 0.85, steps=seq_len * hidden_size, dtype=torch.float32)
    activation = values.reshape(seq_len, hidden_size)
    activation = activation - activation.mean(dim=-1, keepdim=True)
    activation = activation * torch.rsqrt(activation.square().mean(dim=-1, keepdim=True) + 1e-6)
    token_offsets = torch.linspace(-0.05, 0.05, steps=seq_len, dtype=torch.float32).reshape(seq_len, 1)
    activation = activation + token_offsets
    return activation.reshape(1, 1, seq_len, hidden_size).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real DeepSeek V4 Flash attention K/V projection TTNN smoke.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_KV_PROJECTION_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_KV_PROJECTION_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_KV_PROJECTION_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--attn-norm-pcc", type=float, default=0.999)
    parser.add_argument("--kv-linear-pcc", type=float, default=0.99)
    parser.add_argument("--kv-output-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_kv_projection_smoke(
        args.snapshot_dir,
        layer=args.layer,
        seq_len=args.seq_len,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        attn_norm_pcc=args.attn_norm_pcc,
        kv_linear_pcc=args.kv_linear_pcc,
        kv_output_pcc=args.kv_output_pcc,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_kv_projection(
    tensors: Mapping[str, torch.Tensor],
    weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    device_id: int,
) -> dict[str, torch.Tensor]:
    import ttnn

    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_input = ttnn.from_torch(
            activation,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_attn_norm_weight = ttnn.from_torch(
            tensors[f"layers.{layer}.attn_norm.weight"].contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_attn_norm_output = ttnn.rms_norm(
            tt_input,
            weight=tt_attn_norm_weight,
            epsilon=config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_wkv = _to_tt_linear_weight(
            weights.wkv,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_norm_weight = ttnn.from_torch(
            weights.kv_norm.contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_linear = ttnn.linear(tt_attn_norm_output, tt_wkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_kv_output = ttnn.rms_norm(
            tt_kv_linear,
            weight=tt_kv_norm_weight,
            epsilon=config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return {
            "attn_norm_output": ttnn.to_torch(tt_attn_norm_output).contiguous(),
            "kv_linear": ttnn.to_torch(tt_kv_linear).contiguous(),
            "kv_output": ttnn.to_torch(tt_kv_output).contiguous(),
        }
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
    weights: KvProjectionWeights,
    activation: torch.Tensor,
    reference: dict[str, torch.Tensor],
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = _payload_byte_split(metadata)
    return {
        "schema_version": REAL_KV_PROJECTION_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "sequence_length": int(seq_len),
        "model": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "kv_output_dim": _kv_output_dim(config),
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "kv_nope_head_dim": _kv_output_dim(config) - config.qk_rope_head_dim,
            "compress_ratio": config.compress_ratios[layer],
            "sliding_window": config.sliding_window,
            "rms_norm_eps": config.rms_norm_eps,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "projection_scope": {
            "path": "attention_norm -> wkv -> kv_norm",
            "selected_key_contract": {
                "required": [
                    f"layers.{layer}.attn_norm.weight",
                    f"layers.{layer}.attn.kv_norm.weight",
                    f"layers.{layer}.attn.wkv.weight",
                ],
                "conditional": [
                    f"layers.{layer}.attn.wkv.scale when wkv.weight metadata dtype is FP8",
                ],
            },
            "source_layout": "single real-checkpoint wkv projection plus kv_norm",
            "rope_application": "excluded; this smoke returns the pre-RoPE K/V representation",
            "cache_update": "excluded; future prefill/cache code consumes kv_output[:, 0]",
            "not_full_attention": True,
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
        "kv_projection_format": {
            "weight_block_size": list(KV_FP8_BLOCK_SIZE),
            "source_formats": _source_format_summary(tensors, layer=layer),
            "decoded_tensors": {
                "wkv": _tensor_summary(weights.wkv),
                "kv_norm": _tensor_summary(weights.kv_norm),
            },
            "normalization_tensors": {
                "attn_norm": _tensor_summary(tensors[f"layers.{layer}.attn_norm.weight"].to(torch.bfloat16)),
                "kv_norm": _tensor_summary(weights.kv_norm),
            },
        },
        "projection_output_shapes": {
            "attn_norm_output": list(reference["attn_norm_output"].shape),
            "kv_linear": list(reference["kv_linear"].shape),
            "kv_output": list(reference["kv_output"].shape),
            "future_cache_projection": list(reference["cache_projection"].shape),
            "rope_split": {
                "kv_nope_head_dim": _kv_output_dim(config) - config.qk_rope_head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
            },
        },
        "host_boundaries": [
            {
                "name": "kv_fp8_decode_to_bf16",
                "location": "before TTNN linear",
                "description": "FP8 E4M3 wkv weight and UE8M0 block scales are decoded on host to BF16",
            },
            {
                "name": "activation_host_to_device",
                "location": "smoke input",
                "description": "deterministic BF16 activation is generated on host and uploaded to TTNN",
            },
            {
                "name": "projection_output_readback",
                "location": "after kv_norm",
                "description": "TTNN attn_norm, wkv, and kv_norm outputs are copied to host for smoke accuracy",
            },
        ],
        "reference_ops": [
            "decode_kv_projection_weight",
            "torch.rms_norm_reference(attn_norm)",
            "torch.linear(wkv)",
            "torch.rms_norm_reference(kv_norm)",
        ],
        "ttnn_ops": [],
        "inputs": {"activation": _tensor_summary(activation)},
        "reference": {
            "attn_norm_output": _tensor_summary(reference["attn_norm_output"]),
            "kv_linear": _tensor_summary(reference["kv_linear"]),
            "kv_output": _tensor_summary(reference["kv_output"]),
            "cache_projection": _tensor_summary(reference["cache_projection"]),
        },
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _to_tt_linear_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    import ttnn

    torch_weight = weight.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_weight,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _kv_output_dim(config: DeepSeekV4FlashConfig) -> int:
    return int(config.num_key_value_heads) * int(config.head_dim)


def _is_fp8_kv_tensor(weight: torch.Tensor, scale: torch.Tensor | None) -> bool:
    return weight.dtype == torch.float8_e4m3fn and scale is not None and scale.dtype == torch.float8_e8m0fnu


def _metadata_dtype_is_fp8(dtype: str) -> bool:
    return dtype.startswith("F8_")


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {
        "attn_norm": 0,
        "kv_norm": 0,
        "wkv_weight": 0,
        "wkv_scale": 0,
    }
    for item in metadata:
        key = item.canonical_key
        if key.endswith(".attn_norm.weight"):
            split["attn_norm"] += item.nbytes
        elif key.endswith(".attn.kv_norm.weight"):
            split["kv_norm"] += item.nbytes
        elif key.endswith(".attn.wkv.weight"):
            split["wkv_weight"] += item.nbytes
        elif key.endswith(".attn.wkv.scale"):
            split["wkv_scale"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in K/V projection slice: {key}")
    split["norms"] = split["attn_norm"] + split["kv_norm"]
    split["kv_projection"] = split["wkv_weight"] + split["wkv_scale"]
    split["weights"] = split["wkv_weight"]
    split["scales"] = split["wkv_scale"]
    split["total"] = split["norms"] + split["kv_projection"]
    return split


def _source_format_summary(tensors: Mapping[str, torch.Tensor], *, layer: int) -> dict[str, Any]:
    prefix = f"layers.{layer}.attn"
    weight = tensors[f"{prefix}.wkv.weight"]
    scale = tensors.get(f"{prefix}.wkv.scale")
    if _is_fp8_kv_tensor(weight, scale):
        source_format = "FP8_E4M3_WEIGHT_UE8M0_128x128_SCALE"
    elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
        source_format = "DIRECT_FLOAT_WEIGHT_SCALE_NOT_SELECTED"
    else:
        source_format = "UNSUPPORTED"
    return {
        "wkv": {
            "format": source_format,
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "scale_dtype": None if scale is None else str(scale.dtype),
            "scale_shape": None if scale is None else list(scale.shape),
        }
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
    attn_norm_pcc: float,
    kv_linear_pcc: float,
    kv_output_pcc: float,
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in (
        ("attn_norm_pcc", attn_norm_pcc),
        ("kv_linear_pcc", kv_linear_pcc),
        ("kv_output_pcc", kv_output_pcc),
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
