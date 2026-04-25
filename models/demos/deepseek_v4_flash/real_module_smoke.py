# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm, v4_router
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    DEFAULT_LAYER_ROUTER_NORMS_LAYER,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_TENSORS,
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_router_norm_keys,
)

REAL_MODULE_SMOKE_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_NORM_NAME = "attn_norm"
NORM_TENSOR_KEYS = {
    "attn_norm": "attn_norm",
    "ffn_norm": "ffn_norm",
}


def run_real_module_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_ROUTER_NORMS_LAYER,
    norm: str = DEFAULT_NORM_NAME,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    norm_pcc: float = 0.999,
    router_pcc: float = 0.99,
    router_index_match: float = 0.8,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Load one real router/norm slice and optionally execute a tiny TTNN path."""

    _validate_smoke_args(
        norm=norm,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        router_index_match=router_index_match,
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_router_norm_slice(
        snapshot_dir,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    validate_router_norm_slice(tensors, config=config, layer=layer)

    activation = deterministic_activation(
        hidden_size=config.hidden_size,
        seq_len=seq_len,
        gate_weight=tensors[f"layers.{layer}.ffn.gate.weight"],
    )
    reference = build_torch_reference(tensors, config=config, activation=activation, norm=norm)
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        norm=norm,
        seq_len=seq_len,
        config=config,
        metadata=metadata,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        reference=reference,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    if seq_len % 32 != 0:
        raise ValueError(f"TTNN smoke seq_len must be a multiple of 32, got {seq_len}")

    ttnn_outputs = _run_ttnn_slice(
        tensors,
        config=config,
        activation=activation,
        norm=norm,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = ["ttnn.rms_norm", "TtRouter(ttnn.linear+host_topk)"]
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
    }
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def load_router_norm_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_router_norm_keys(index, layer=layer)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def validate_router_norm_slice(
    tensors: dict[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> None:
    prefix = f"layers.{layer}"
    expected_shapes = {
        f"{prefix}.attn_norm.weight": (config.hidden_size,),
        f"{prefix}.ffn_norm.weight": (config.hidden_size,),
        f"{prefix}.ffn.gate.weight": (config.n_routed_experts, config.hidden_size),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in tensors:
            raise KeyError(f"Missing required real smoke tensor {key!r}")
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


def deterministic_activation(
    *,
    hidden_size: int,
    seq_len: int,
    gate_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    if gate_weight is not None:
        if gate_weight.ndim != 2 or gate_weight.shape[-1] != hidden_size:
            raise ValueError(f"gate_weight must have shape [experts, {hidden_size}], got {tuple(gate_weight.shape)}")
        rows = gate_weight[torch.arange(seq_len, dtype=torch.long) % gate_weight.shape[0]].float()
        scales = torch.linspace(0.95, 1.05, steps=seq_len, dtype=torch.float32).reshape(seq_len, 1)
        return (rows * scales).reshape(1, 1, seq_len, hidden_size).to(torch.bfloat16)

    values = torch.linspace(-0.35, 0.45, steps=seq_len * hidden_size, dtype=torch.float32)
    token_offsets = torch.linspace(-0.03, 0.03, steps=seq_len, dtype=torch.float32).repeat_interleave(hidden_size)
    activation = (values + token_offsets).reshape(1, 1, seq_len, hidden_size)
    return activation.to(torch.bfloat16)


def build_torch_reference(
    tensors: dict[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    norm: str,
) -> dict[str, torch.Tensor | None]:
    _validate_activation(activation, hidden_size=config.hidden_size)
    layer = _infer_layer(tensors)
    norm_weight = tensors[_norm_key(layer, norm)].to(torch.bfloat16)
    gate_weight = tensors[f"layers.{layer}.ffn.gate.weight"].to(torch.bfloat16)
    bias = tensors.get(f"layers.{layer}.ffn.gate.bias")
    tid2eid = tensors.get(f"layers.{layer}.ffn.gate.tid2eid")
    input_ids = _deterministic_input_ids(config.vocab_size, activation.shape[-2]) if tid2eid is not None else None

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
    return {
        "norm_output": norm_output,
        "router_weights": router_weights,
        "router_indices": router_indices,
        "input_ids": input_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny DeepSeek V4 Flash real-checkpoint TTNN module smoke path.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_ROUTER_NORMS_LAYER)
    parser.add_argument("--norm", choices=sorted(NORM_TENSOR_KEYS), default=DEFAULT_NORM_NAME)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--norm-pcc", type=float, default=0.999)
    parser.add_argument("--router-pcc", type=float, default=0.99)
    parser.add_argument("--router-index-match", type=float, default=0.8)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_module_smoke(
        args.snapshot_dir,
        layer=args.layer,
        norm=args.norm,
        seq_len=args.seq_len,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        norm_pcc=args.norm_pcc,
        router_pcc=args.router_pcc,
        router_index_match=args.router_index_match,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_slice(
    tensors: dict[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    norm: str,
    device_id: int,
) -> dict[str, torch.Tensor]:
    import ttnn
    from models.demos.deepseek_v4_flash.ttnn_router import TtRouter

    layer = _infer_layer(tensors)
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
            tensors[_norm_key(layer, norm)].contiguous().to(torch.bfloat16),
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
            gate_weight=tensors[f"layers.{layer}.ffn.gate.weight"],
            bias=tensors.get(f"layers.{layer}.ffn.gate.bias"),
            tid2eid=tensors.get(f"layers.{layer}.ffn.gate.tid2eid"),
            topk=config.num_experts_per_tok,
            route_scale=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_ids = _deterministic_input_ids(config.vocab_size, activation.shape[-2])
        if tensors.get(f"layers.{layer}.ffn.gate.tid2eid") is None:
            input_ids = None
        router_weights, router_indices = router(tt_norm, input_ids=input_ids)
        return {
            "norm_output": ttnn.to_torch(tt_norm).contiguous(),
            "router_weights": router_weights.contiguous(),
            "router_indices": router_indices.contiguous(),
        }
    finally:
        ttnn.close_device(device)


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    norm: str,
    seq_len: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    max_tensors: int,
    max_bytes: int,
    reference: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    return {
        "schema_version": REAL_MODULE_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "norm": norm,
        "sequence_length": int(seq_len),
        "model": {
            "hidden_size": config.hidden_size,
            "n_routed_experts": config.n_routed_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "scoring_func": config.scoring_func,
            "routed_scaling_factor": config.routed_scaling_factor,
            "rms_norm_eps": config.rms_norm_eps,
        },
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "payload_bytes": int(sum(item.nbytes for item in metadata)),
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": int(sum(item.nbytes for item in metadata)),
        },
        "reference_ops": ["torch.rms_norm_reference", "torch.router_reference"],
        "ttnn_ops": [],
        "reference": {
            "rms_norm": _tensor_summary(reference["norm_output"]),
            "router_weights": _tensor_summary(reference["router_weights"]),
            "router_indices": _tensor_summary(reference["router_indices"]),
            "input_ids": _tensor_summary(reference["input_ids"]),
        },
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
    passed = pcc >= pcc_threshold and allclose
    return {
        "passed": bool(passed),
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
    norm: str,
    seq_len: int,
    max_tensors: int,
    max_bytes: int,
    router_index_match: float,
) -> None:
    if norm not in NORM_TENSOR_KEYS:
        raise ValueError(f"norm must be one of {sorted(NORM_TENSOR_KEYS)}, got {norm!r}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if not 0.0 <= router_index_match <= 1.0:
        raise ValueError(f"router_index_match must be in [0, 1], got {router_index_match}")


def _validate_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim != 4 or activation.shape[:2] != (1, 1):
        raise ValueError(f"activation must have shape [1, 1, tokens, hidden], got {tuple(activation.shape)}")
    if activation.shape[-1] != hidden_size:
        raise ValueError(f"activation hidden size must be {hidden_size}, got {activation.shape[-1]}")


def _infer_layer(tensors: dict[str, torch.Tensor]) -> int:
    layers = set()
    for key in tensors:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "layers":
            layers.add(int(parts[1]))
    if len(layers) != 1:
        raise ValueError(f"Expected tensors from exactly one layer, got {sorted(layers)}")
    return layers.pop()


def _norm_key(layer: int, norm: str) -> str:
    return f"layers.{layer}.{NORM_TENSOR_KEYS[norm]}.weight"


def _deterministic_input_ids(vocab_size: int, seq_len: int) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return (torch.arange(seq_len, dtype=torch.int64) % vocab_size).reshape(1, seq_len)


if __name__ == "__main__":
    main()
