# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from safetensors import safe_open

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest


PlacementStrategy = Literal["tp_shard_replicate_ep", "expert_home_device", "replicate_all", "host_metadata"]

_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\.(?P<kind>weight_packed|scale)$"
)
_LAYER_RE = re.compile(r"^layers\.(?P<layer>\d+)\.")
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}


@dataclass(frozen=True)
class WeightTensorRecord:
    key: str
    artifact: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    category: Literal["non_expert", "expert", "metadata"]
    layer: int | None = None
    expert: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "artifact": self.artifact,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "category": self.category,
            "layer": self.layer,
            "expert": self.expert,
        }


@dataclass(frozen=True)
class WeightPlacement:
    key: str
    strategy: PlacementStrategy
    devices: tuple[tuple[int, int], ...]
    per_device_nbytes: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "strategy": self.strategy,
            "devices": [list(coord) for coord in self.devices],
            "per_device_nbytes": self.per_device_nbytes,
        }


@dataclass(frozen=True)
class WeightInventoryReport:
    preprocessed_dir: str
    mesh_shape: tuple[int, int]
    tensor_count: int
    total_nbytes: int
    counts_by_category: Mapping[str, int]
    counts_by_strategy: Mapping[str, int]
    per_device_nbytes: Mapping[tuple[int, int], int]

    @property
    def max_device_weight_nbytes(self) -> int:
        return max(self.per_device_nbytes.values(), default=0)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "preprocessed_dir": self.preprocessed_dir,
            "mesh_shape": list(self.mesh_shape),
            "tensor_count": self.tensor_count,
            "total_nbytes": self.total_nbytes,
            "counts_by_category": dict(self.counts_by_category),
            "counts_by_strategy": dict(self.counts_by_strategy),
            "per_device_nbytes": {f"{row},{col}": nbytes for (row, col), nbytes in self.per_device_nbytes.items()},
            "max_device_weight_nbytes": self.max_device_weight_nbytes,
        }


def build_weight_inventory_report(
    preprocessed_dir: str | Path,
    *,
    mesh_shape: tuple[int, int] = (2, 4),
    large_tensor_threshold: int = 1 << 20,
) -> WeightInventoryReport:
    records = read_weight_tensor_records(preprocessed_dir)
    placements = plan_weight_placements(
        records,
        mesh_shape=mesh_shape,
        large_tensor_threshold=large_tensor_threshold,
    )
    per_device = {coord: 0 for coord in _mesh_coords(mesh_shape)}
    for placement in placements:
        for coord in placement.devices:
            per_device[coord] += int(placement.per_device_nbytes)
    return WeightInventoryReport(
        preprocessed_dir=str(Path(preprocessed_dir).expanduser().resolve()),
        mesh_shape=tuple(mesh_shape),
        tensor_count=len(records),
        total_nbytes=sum(record.nbytes for record in records),
        counts_by_category=Counter(record.category for record in records),
        counts_by_strategy=Counter(placement.strategy for placement in placements),
        per_device_nbytes=per_device,
    )


def read_weight_tensor_records(preprocessed_dir: str | Path) -> tuple[WeightTensorRecord, ...]:
    preprocessed_dir = Path(preprocessed_dir).expanduser().resolve()
    manifest = load_tt_manifest(preprocessed_dir)
    records: list[WeightTensorRecord] = []
    for category, artifact_field in (
        ("non_expert", "non_expert_safetensors"),
        ("expert", "expert_safetensors"),
    ):
        for artifact in manifest["artifacts"][artifact_field]:
            records.extend(_read_artifact_records(preprocessed_dir, str(artifact), category=category))
    metadata_artifact = manifest["artifacts"]["metadata_safetensors"]
    records.extend(_read_artifact_records(preprocessed_dir, str(metadata_artifact), category="metadata"))
    return tuple(sorted(records, key=lambda record: record.key))


def plan_weight_placements(
    records: Sequence[WeightTensorRecord],
    *,
    mesh_shape: tuple[int, int],
    large_tensor_threshold: int = 1 << 20,
) -> tuple[WeightPlacement, ...]:
    _validate_mesh_shape(mesh_shape)
    tp = mesh_shape[1]
    devices = _mesh_coords(mesh_shape)
    placements: list[WeightPlacement] = []
    for record in records:
        if record.category == "metadata":
            placements.append(
                WeightPlacement(
                    key=record.key,
                    strategy="host_metadata",
                    devices=(),
                    per_device_nbytes=0,
                )
            )
        elif record.category == "expert":
            device = _expert_home_device(record, mesh_shape)
            placements.append(
                WeightPlacement(
                    key=record.key,
                    strategy="expert_home_device",
                    devices=(device,),
                    per_device_nbytes=record.nbytes,
                )
            )
        elif _is_large_shardable_non_expert(record, tp=tp, threshold=large_tensor_threshold):
            placements.append(
                WeightPlacement(
                    key=record.key,
                    strategy="tp_shard_replicate_ep",
                    devices=devices,
                    per_device_nbytes=_ceil_div(record.nbytes, tp),
                )
            )
        else:
            placements.append(
                WeightPlacement(
                    key=record.key,
                    strategy="replicate_all",
                    devices=devices,
                    per_device_nbytes=record.nbytes,
                )
            )
    return tuple(placements)


def estimate_decode_cache_nbytes(
    config: DeepSeekV4FlashConfig,
    *,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
    total_values = 0
    for ratio in config.compress_ratios[: config.num_hidden_layers]:
        total_values += config.sliding_window * config.head_dim
        if ratio:
            compressed_len = seq_len // int(ratio)
            total_values += compressed_len * config.head_dim
            if int(ratio) == 4:
                total_values += compressed_len * config.index_head_dim
    return int(total_values * batch_size * dtype_bytes)


def estimate_max_seq_len_supported(
    config: DeepSeekV4FlashConfig,
    inventory: WeightInventoryReport,
    *,
    device_dram_bytes: int,
    cache_dtype_bytes: int = 2,
    safety_margin_bytes: int = 1 << 30,
) -> int:
    if device_dram_bytes <= 0:
        raise ValueError(f"device_dram_bytes must be positive, got {device_dram_bytes}")
    budget = int(device_dram_bytes) - int(safety_margin_bytes) - int(inventory.max_device_weight_nbytes)
    if budget <= 0:
        return 0
    configured = _configured_max_position_embeddings(config)
    if estimate_decode_cache_nbytes(config, seq_len=configured, dtype_bytes=cache_dtype_bytes) <= budget:
        return configured

    low, high = 0, configured
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_decode_cache_nbytes(config, seq_len=mid, dtype_bytes=cache_dtype_bytes) <= budget:
            low = mid
        else:
            high = mid - 1
    return int(low)


def _read_artifact_records(
    preprocessed_dir: Path,
    artifact: str,
    *,
    category: Literal["non_expert", "expert", "metadata"],
) -> list[WeightTensorRecord]:
    path = preprocessed_dir / artifact
    records: list[WeightTensorRecord] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
            dtype = str(handle.get_slice(key).get_dtype())
            expert_match = _EXPERT_RE.match(key)
            layer_match = _LAYER_RE.match(key)
            records.append(
                WeightTensorRecord(
                    key=str(key),
                    artifact=artifact,
                    dtype=dtype,
                    shape=shape,
                    nbytes=_safetensors_nbytes(dtype, shape),
                    category=category,
                    layer=int(expert_match.group("layer"))
                    if expert_match is not None
                    else int(layer_match.group("layer"))
                    if layer_match is not None
                    else None,
                    expert=int(expert_match.group("expert")) if expert_match is not None else None,
                )
            )
    return records


def _is_large_shardable_non_expert(record: WeightTensorRecord, *, tp: int, threshold: int) -> bool:
    if record.nbytes < threshold or len(record.shape) < 2:
        return False
    return any(dim % tp == 0 for dim in record.shape)


def _expert_home_device(record: WeightTensorRecord, mesh_shape: tuple[int, int]) -> tuple[int, int]:
    if record.expert is None:
        raise ValueError(f"Expert record missing expert id: {record.key}")
    rows, cols = mesh_shape
    device_index = int(record.expert) % (rows * cols)
    return device_index // cols, device_index % cols


def _configured_max_position_embeddings(config: DeepSeekV4FlashConfig) -> int:
    value = config.rope_scaling.get("factor")
    original = config.rope_scaling.get("original_max_position_embeddings")
    if isinstance(value, (int, float)) and isinstance(original, int):
        return int(original * value)
    return 1048576


def _safetensors_nbytes(dtype: str, shape: Sequence[int]) -> int:
    numel = math.prod(shape)
    if dtype in _SAFETENSORS_DTYPE_BYTES:
        return int(numel * _SAFETENSORS_DTYPE_BYTES[dtype])
    if dtype.startswith("F4"):
        return int((numel + 1) // 2)
    raise ValueError(f"Unsupported safetensors dtype for budget accounting: {dtype}")


def _mesh_coords(mesh_shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    _validate_mesh_shape(mesh_shape)
    rows, cols = mesh_shape
    return tuple((row, col) for row in range(rows) for col in range(cols))


def _validate_mesh_shape(mesh_shape: tuple[int, int]) -> None:
    if len(mesh_shape) != 2:
        raise ValueError(f"mesh_shape must be (rows, cols), got {mesh_shape}")
    rows, cols = mesh_shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"mesh_shape dimensions must be positive, got {mesh_shape}")


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)
