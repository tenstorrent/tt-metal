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
import torch.nn.functional as F

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, TensorMetadata
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_fp8_block_scaled_weight
from models.demos.deepseek_v4_flash.ttnn_attention_projection import AttentionProjectionWeights, TtAttentionProjection

REAL_ATTENTION_PROJECTION_SMOKE_SCHEMA_VERSION = 1
DEFAULT_LAYER_ATTENTION_PROJECTION_LAYER = 3
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS = 6
DEFAULT_ATTENTION_PROJECTION_MAX_BYTES = 48 * 1024 * 1024
ATTENTION_FP8_BLOCK_SIZE = (128, 128)
ATTENTION_TTNN_TILE_MULTIPLE = 32
ATTENTION_Q_PROJECTIONS = ("wq_a", "wq_b")
_DIRECT_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def run_real_attention_projection_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_ATTENTION_PROJECTION_LAYER,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS,
    max_bytes: int = DEFAULT_ATTENTION_PROJECTION_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    attn_norm_pcc: float = 0.999,
    q_rank_pcc: float = 0.999,
    q_output_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Load and execute one real DeepSeek V4 Flash attention query projection slice."""

    _validate_smoke_args(
        layer=layer,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        attn_norm_pcc=attn_norm_pcc,
        q_rank_pcc=q_rank_pcc,
        q_output_pcc=q_output_pcc,
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_real_attention_projection_slice(
        snapshot_dir,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    weights = decode_real_attention_projection_weights(tensors, config=config, layer=layer)

    activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    reference = build_torch_attention_projection_reference(
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

    if seq_len % ATTENTION_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN smoke seq_len must be a multiple of {ATTENTION_TTNN_TILE_MULTIPLE}, got {seq_len}")

    ttnn_outputs = _run_ttnn_attention_projection(
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
        "TtAttentionProjection.project_q_rank",
        "ttnn.linear(wq_a)",
        "ttnn.rms_norm(q_norm)",
        "TtAttentionProjection.project_q_from_rank",
        "ttnn.linear(wq_b)",
    ]
    result["ttnn"] = {
        "attn_norm_output": _tensor_summary(ttnn_outputs["attn_norm_output"]),
        "q_rank_norm": _tensor_summary(ttnn_outputs["q_rank_norm"]),
        "q_output": _tensor_summary(ttnn_outputs["q_output"]),
    }
    result["accuracy"] = {
        "attn_norm_output": _accuracy_summary(
            reference["attn_norm_output"],
            ttnn_outputs["attn_norm_output"],
            pcc_threshold=attn_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "q_rank_norm": _accuracy_summary(
            reference["q_rank_norm"],
            ttnn_outputs["q_rank_norm"],
            pcc_threshold=q_rank_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "q_output": _accuracy_summary(
            reference["q_output"],
            ttnn_outputs["q_output"],
            pcc_threshold=q_output_pcc,
            rtol=rtol,
            atol=atol,
        ),
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_attention_projection_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")

    prefix = f"layers.{layer}.attn"
    required_keys = [
        f"layers.{layer}.attn_norm.weight",
        f"{prefix}.q_norm.weight",
    ]
    for key in required_keys:
        index.location(key)

    projection_weight_keys = [f"{prefix}.{projection}.weight" for projection in ATTENTION_Q_PROJECTIONS]
    projection_metadata = {item.canonical_key: item for item in index.metadata_for_keys(projection_weight_keys)}

    keys = list(required_keys)
    for projection, weight_key in zip(ATTENTION_Q_PROJECTIONS, projection_weight_keys):
        keys.append(weight_key)
        if _metadata_dtype_is_fp8(projection_metadata[weight_key].dtype):
            scale_key = f"{prefix}.{projection}.scale"
            index.location(scale_key)
            keys.append(scale_key)
    return keys


def load_real_attention_projection_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    max_tensors: int = DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS,
    max_bytes: int = DEFAULT_ATTENTION_PROJECTION_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_attention_projection_keys(index, layer=layer)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def decode_real_attention_projection_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    dtype: torch.dtype = torch.bfloat16,
) -> AttentionProjectionWeights:
    validate_real_attention_projection_slice(tensors, config=config, layer=layer)
    prefix = f"layers.{layer}.attn"
    decoded: dict[str, torch.Tensor] = {}
    for projection in ATTENTION_Q_PROJECTIONS:
        weight = tensors[f"{prefix}.{projection}.weight"]
        scale = tensors.get(f"{prefix}.{projection}.scale")
        if _is_fp8_attention_tensor(weight, scale):
            decoded[projection] = decode_fp8_block_scaled_weight(
                weight,
                scale,
                block_size=ATTENTION_FP8_BLOCK_SIZE,
                dtype=dtype,
            )
        elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
            decoded[projection] = weight.contiguous().to(dtype)
        else:
            scale_dtype = None if scale is None else scale.dtype
            raise TypeError(
                f"Unsupported attention projection format for {prefix}.{projection}: "
                f"weight dtype {weight.dtype}, scale dtype {scale_dtype}"
            )
    return AttentionProjectionWeights(
        wq_a=decoded["wq_a"],
        q_norm=tensors[f"{prefix}.q_norm.weight"].contiguous().to(dtype),
        wq_b=decoded["wq_b"],
    )


def validate_real_attention_projection_slice(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> None:
    hidden = int(config.hidden_size)
    q_lora_rank = int(config.q_lora_rank)
    q_output_dim = int(config.num_attention_heads) * int(config.head_dim)
    prefix = f"layers.{layer}.attn"
    expected_shapes = {
        f"layers.{layer}.attn_norm.weight": (hidden,),
        f"{prefix}.q_norm.weight": (q_lora_rank,),
        f"{prefix}.wq_a.weight": (q_lora_rank, hidden),
        f"{prefix}.wq_b.weight": (q_output_dim, q_lora_rank),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in tensors:
            raise KeyError(f"Missing required real attention projection tensor {key!r}")
        if tuple(tensors[key].shape) != expected_shape:
            raise ValueError(f"Expected {key} shape {expected_shape}, got {tuple(tensors[key].shape)}")

    for projection in ATTENTION_Q_PROJECTIONS:
        weight_key = f"{prefix}.{projection}.weight"
        scale_key = f"{prefix}.{projection}.scale"
        weight = tensors[weight_key]
        scale = tensors.get(scale_key)
        if _is_fp8_attention_tensor(weight, scale):
            expected_scale_shape = (
                math.ceil(weight.shape[0] / ATTENTION_FP8_BLOCK_SIZE[0]),
                math.ceil(weight.shape[1] / ATTENTION_FP8_BLOCK_SIZE[1]),
            )
            if tuple(scale.shape) != expected_scale_shape:
                raise ValueError(f"Expected {scale_key} shape {expected_scale_shape}, got {tuple(scale.shape)}")
        elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
            if scale is not None and scale.ndim != 2:
                raise ValueError(f"Expected {scale_key} to be rank 2, got {tuple(scale.shape)}")
        else:
            scale_dtype = None if scale is None else scale.dtype
            raise TypeError(
                f"Unsupported attention projection format for {prefix}.{projection}: "
                f"weight dtype {weight.dtype}, scale dtype {scale_dtype}"
            )


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


def build_torch_attention_projection_reference(
    tensors: Mapping[str, torch.Tensor],
    weights: AttentionProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_activation(activation, hidden_size=config.hidden_size)
    prefix = f"layers.{layer}.attn"
    attn_norm_weight = tensors[f"layers.{layer}.attn_norm.weight"].to(torch.bfloat16)
    attn_norm_output = rms_norm(activation[:, 0], attn_norm_weight, eps=config.rms_norm_eps).unsqueeze(1)
    q_rank_linear = F.linear(attn_norm_output[:, 0].float(), weights.wq_a.float()).to(torch.bfloat16)
    q_rank_norm = rms_norm(q_rank_linear, weights.q_norm, config.rms_norm_eps).unsqueeze(1)
    q_output = F.linear(q_rank_norm[:, 0].float(), weights.wq_b.float()).unsqueeze(1)
    return {
        "attn_norm_output": attn_norm_output,
        "q_rank_linear": q_rank_linear.unsqueeze(1),
        "q_rank_norm": q_rank_norm,
        "q_output": q_output,
        "attn_norm_weight": attn_norm_weight,
        "q_norm_weight": tensors[f"{prefix}.q_norm.weight"].to(torch.bfloat16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real DeepSeek V4 Flash attention query projection TTNN smoke path."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_ATTENTION_PROJECTION_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_ATTENTION_PROJECTION_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--attn-norm-pcc", type=float, default=0.999)
    parser.add_argument("--q-rank-pcc", type=float, default=0.999)
    parser.add_argument("--q-output-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_attention_projection_smoke(
        args.snapshot_dir,
        layer=args.layer,
        seq_len=args.seq_len,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        attn_norm_pcc=args.attn_norm_pcc,
        q_rank_pcc=args.q_rank_pcc,
        q_output_pcc=args.q_output_pcc,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_attention_projection(
    tensors: Mapping[str, torch.Tensor],
    weights: AttentionProjectionWeights,
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
        module = TtAttentionProjection(
            device=device,
            weights=weights,
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            norm_eps=config.rms_norm_eps,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q_rank_norm = module.project_q_rank(tt_attn_norm_output)
        tt_q_output = module.project_q_from_rank(tt_q_rank_norm)
        return {
            "attn_norm_output": ttnn.to_torch(tt_attn_norm_output).contiguous(),
            "q_rank_norm": ttnn.to_torch(tt_q_rank_norm).contiguous(),
            "q_output": ttnn.to_torch(tt_q_output).contiguous(),
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
    weights: AttentionProjectionWeights,
    activation: torch.Tensor,
    reference: dict[str, torch.Tensor],
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = _payload_byte_split(metadata)
    return {
        "schema_version": REAL_ATTENTION_PROJECTION_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "sequence_length": int(seq_len),
        "model": {
            "hidden_size": config.hidden_size,
            "q_lora_rank": config.q_lora_rank,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "projection_scope": {
            "path": "attention_norm -> wq_a -> q_norm -> wq_b",
            "output_projection": "excluded",
            "output_projection_exclusion_reason": (
                "wo_a and wo_b are separate FP8 block-scaled tensors of about 64 MiB total payload "
                "and wo_a requires grouped output-projection handling outside this query smoke"
            ),
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
        "attention_projection_format": {
            "weight_block_size": list(ATTENTION_FP8_BLOCK_SIZE),
            "source_formats": _source_format_summary(tensors, layer=layer),
            "decoded_tensors": {
                "wq_a": _tensor_summary(weights.wq_a),
                "q_norm": _tensor_summary(weights.q_norm),
                "wq_b": _tensor_summary(weights.wq_b),
            },
            "normalization_tensors": {
                "attn_norm": _tensor_summary(tensors[f"layers.{layer}.attn_norm.weight"].to(torch.bfloat16)),
                "q_norm": _tensor_summary(weights.q_norm),
            },
        },
        "host_boundaries": [
            {
                "name": "attention_fp8_decode_to_bf16",
                "location": "before TtAttentionProjection",
                "description": "FP8 E4M3 query weights and UE8M0 block scales are decoded on host to BF16",
            },
            {
                "name": "activation_host_to_device",
                "location": "smoke input",
                "description": "deterministic BF16 activation is generated on host and uploaded to TTNN",
            },
            {
                "name": "projection_output_readback",
                "location": "after query projection",
                "description": "TTNN attn_norm, q_rank_norm, and q_output tensors are copied to host for smoke accuracy",
            },
        ],
        "reference_ops": [
            "decode_attention_projection_weights",
            "torch.rms_norm_reference(attn_norm)",
            "torch.linear(wq_a)",
            "torch.rms_norm_reference(q_norm)",
            "torch.linear(wq_b)",
        ],
        "ttnn_ops": [],
        "inputs": {"activation": _tensor_summary(activation)},
        "reference": {
            "attn_norm_output": _tensor_summary(reference["attn_norm_output"]),
            "q_rank_linear": _tensor_summary(reference["q_rank_linear"]),
            "q_rank_norm": _tensor_summary(reference["q_rank_norm"]),
            "q_output": _tensor_summary(reference["q_output"]),
        },
        "ttnn": {},
        "accuracy": {},
        "passed": False,
    }


def _is_fp8_attention_tensor(weight: torch.Tensor, scale: torch.Tensor | None) -> bool:
    return weight.dtype == torch.float8_e4m3fn and scale is not None and scale.dtype == torch.float8_e8m0fnu


def _metadata_dtype_is_fp8(dtype: str) -> bool:
    return dtype.startswith("F8_")


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {
        "attn_norm": 0,
        "q_norm": 0,
        "wq_a_weight": 0,
        "wq_a_scale": 0,
        "wq_b_weight": 0,
        "wq_b_scale": 0,
    }
    for item in metadata:
        key = item.canonical_key
        if key.endswith(".attn_norm.weight"):
            split["attn_norm"] += item.nbytes
        elif key.endswith(".attn.q_norm.weight"):
            split["q_norm"] += item.nbytes
        elif key.endswith(".attn.wq_a.weight"):
            split["wq_a_weight"] += item.nbytes
        elif key.endswith(".attn.wq_a.scale"):
            split["wq_a_scale"] += item.nbytes
        elif key.endswith(".attn.wq_b.weight"):
            split["wq_b_weight"] += item.nbytes
        elif key.endswith(".attn.wq_b.scale"):
            split["wq_b_scale"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in attention projection slice: {key}")
    split["norms"] = split["attn_norm"] + split["q_norm"]
    split["q_low_rank"] = split["wq_a_weight"] + split["wq_a_scale"]
    split["q_output"] = split["wq_b_weight"] + split["wq_b_scale"]
    split["weights"] = split["wq_a_weight"] + split["wq_b_weight"]
    split["scales"] = split["wq_a_scale"] + split["wq_b_scale"]
    split["total"] = split["norms"] + split["q_low_rank"] + split["q_output"]
    return split


def _source_format_summary(tensors: Mapping[str, torch.Tensor], *, layer: int) -> dict[str, Any]:
    formats = {}
    prefix = f"layers.{layer}.attn"
    for projection in ATTENTION_Q_PROJECTIONS:
        weight = tensors[f"{prefix}.{projection}.weight"]
        scale = tensors.get(f"{prefix}.{projection}.scale")
        if _is_fp8_attention_tensor(weight, scale):
            source_format = "FP8_E4M3_WEIGHT_UE8M0_128x128_SCALE"
        elif weight.dtype in _DIRECT_WEIGHT_DTYPES:
            source_format = "DIRECT_FLOAT_WEIGHT_SCALE_NOT_SELECTED"
        else:
            source_format = "UNSUPPORTED"
        formats[projection] = {
            "format": source_format,
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "scale_dtype": None if scale is None else str(scale.dtype),
            "scale_shape": None if scale is None else list(scale.shape),
        }
    return formats


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
    q_rank_pcc: float,
    q_output_pcc: float,
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
        ("q_rank_pcc", q_rank_pcc),
        ("q_output_pcc", q_output_pcc),
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
