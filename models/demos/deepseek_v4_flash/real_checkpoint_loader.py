# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v4_flash.converter import MODEL_INDEX_FILENAME
from models.demos.deepseek_v4_flash.key_mapping import normalize_hf_key
from models.demos.deepseek_v4_flash.manifest import MODEL_NAME

SELECTIVE_MATERIALIZATION_SCHEMA_VERSION = 1
DEFAULT_LAYER_ROUTER_NORMS_LAYER = 3
DEFAULT_LAYER_EXPERT_MLP_LAYER = 3
DEFAULT_LAYER_EXPERT_MLP_EXPERT = 0
DEFAULT_MAX_TENSORS = 8
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
LAYER_ROUTER_NORMS_SELECTOR = "layer-router-norms"
SLICE_ARTIFACT_FILENAME = "selected_tensors.safetensors"
SLICE_MANIFEST_FILENAME = "materialized_slice_manifest.json"
EXPERT_MLP_PROJECTIONS = ("w1", "w2", "w3")

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
class TensorLocation:
    canonical_key: str
    source_key: str
    shard_name: str
    shard_path: Path


@dataclass(frozen=True)
class TensorMetadata:
    canonical_key: str
    source_key: str
    shard_name: str
    shard_path: Path
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    def to_manifest_dict(self, snapshot_dir: Path) -> dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "source_key": self.source_key,
            "shard": self.shard_name,
            "shard_path": _relative_or_absolute(self.shard_path, snapshot_dir),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
        }


class RealCheckpointTensorIndex:
    """Selective metadata and payload loader for a real HF safetensors snapshot."""

    def __init__(self, snapshot_dir: str | Path, locations_by_canonical: dict[str, TensorLocation]):
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve()
        self._locations_by_canonical = dict(locations_by_canonical)

    @classmethod
    def from_snapshot(cls, snapshot_dir: str | Path) -> "RealCheckpointTensorIndex":
        snapshot_dir = Path(snapshot_dir).expanduser().resolve()
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"HF snapshot directory does not exist: {snapshot_dir}")

        index_path = snapshot_dir / MODEL_INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing required {MODEL_INDEX_FILENAME}: {index_path}")

        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Expected non-empty weight_map in {index_path}")

        locations_by_canonical: dict[str, TensorLocation] = {}
        for source_key, shard_name in weight_map.items():
            source_key = str(source_key)
            shard_name = str(shard_name)
            canonical_key = normalize_hf_key(source_key).canonical
            shard_path = snapshot_dir / shard_name
            location = TensorLocation(
                canonical_key=canonical_key,
                source_key=source_key,
                shard_name=shard_name,
                shard_path=shard_path,
            )
            existing = locations_by_canonical.get(canonical_key)
            if existing is not None and existing != location:
                raise ValueError(
                    "Multiple source tensors normalize to "
                    f"{canonical_key!r}: {existing.source_key!r} and {source_key!r}"
                )
            locations_by_canonical[canonical_key] = location

        return cls(snapshot_dir, locations_by_canonical)

    def has_tensor(self, canonical_key: str) -> bool:
        return canonical_key in self._locations_by_canonical

    def location(self, canonical_key: str) -> TensorLocation:
        try:
            return self._locations_by_canonical[canonical_key]
        except KeyError as exc:
            raise KeyError(f"Tensor {canonical_key!r} is not present in {self.snapshot_dir}") from exc

    def metadata_for_keys(self, canonical_keys: Iterable[str]) -> list[TensorMetadata]:
        locations = self._locations_for_keys(canonical_keys)
        return _read_selected_metadata(self.snapshot_dir, locations)

    def load_tensors(
        self,
        canonical_keys: Iterable[str],
        *,
        max_tensors: int = DEFAULT_MAX_TENSORS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
        metadata = self.metadata_for_keys(canonical_keys)
        _enforce_budget(metadata, max_tensors=max_tensors, max_bytes=max_bytes)
        tensors: dict[str, torch.Tensor] = {}
        for shard_name, shard_metadata in _group_metadata_by_shard(metadata).items():
            shard_path = self.snapshot_dir / shard_name
            _raise_if_missing_shard(shard_path)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for item in shard_metadata:
                    tensors[item.canonical_key] = handle.get_tensor(item.source_key).contiguous()
        return tensors, metadata

    def _locations_for_keys(self, canonical_keys: Iterable[str]) -> list[TensorLocation]:
        keys = list(canonical_keys)
        if not keys:
            raise ValueError("At least one tensor key must be requested")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Tensor keys must be unique, got {keys}")
        return [self.location(key) for key in keys]


def layer_router_norm_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    prefix = f"layers.{layer}"
    keys = [
        f"{prefix}.attn_norm.weight",
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


def layer_expert_mlp_keys(index: RealCheckpointTensorIndex, *, layer: int, expert: int) -> list[str]:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if expert < 0:
        raise ValueError(f"expert must be non-negative, got {expert}")

    prefix = f"layers.{layer}.ffn.experts.{expert}"
    keys = [f"{prefix}.{projection}.{kind}" for projection in EXPERT_MLP_PROJECTIONS for kind in ("weight", "scale")]
    for key in keys:
        index.location(key)
    return keys


def materialize_layer_router_norm_slice(
    snapshot_dir: str | Path,
    output_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_ROUTER_NORMS_LAYER,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    overwrite: bool = False,
) -> Path:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_router_norm_keys(index, layer=layer)
    tensors, metadata = index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_dir}")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True)

    artifact_path = output_dir / SLICE_ARTIFACT_FILENAME
    save_file({key: tensors[key] for key in keys}, str(artifact_path))

    manifest = _build_slice_manifest(
        index=index,
        layer=layer,
        keys=keys,
        metadata=metadata,
        artifact_path=artifact_path,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    _write_json(output_dir / SLICE_MANIFEST_FILENAME, manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selectively materialize a tiny DeepSeek V4 Flash real-checkpoint tensor slice."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selector", choices=(LAYER_ROUTER_NORMS_SELECTOR,), default=LAYER_ROUTER_NORMS_SELECTOR)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_ROUTER_NORMS_LAYER)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.selector != LAYER_ROUTER_NORMS_SELECTOR:
        raise ValueError(f"Unsupported selector {args.selector!r}")

    output_dir = materialize_layer_router_norm_slice(
        args.snapshot_dir,
        args.output_dir,
        layer=args.layer,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / SLICE_MANIFEST_FILENAME)}))


def _read_selected_metadata(snapshot_dir: Path, locations: Sequence[TensorLocation]) -> list[TensorMetadata]:
    metadata: list[TensorMetadata] = []
    for shard_name, shard_locations in _group_locations_by_shard(locations).items():
        shard_path = snapshot_dir / shard_name
        _raise_if_missing_shard(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for location in shard_locations:
                slice_handle = handle.get_slice(location.source_key)
                dtype = str(slice_handle.get_dtype())
                shape = tuple(int(dim) for dim in slice_handle.get_shape())
                metadata.append(
                    TensorMetadata(
                        canonical_key=location.canonical_key,
                        source_key=location.source_key,
                        shard_name=location.shard_name,
                        shard_path=location.shard_path,
                        dtype=dtype,
                        shape=shape,
                        nbytes=_safetensors_nbytes(dtype, shape),
                    )
                )
    return metadata


def _build_slice_manifest(
    *,
    index: RealCheckpointTensorIndex,
    layer: int,
    keys: Sequence[str],
    metadata: Sequence[TensorMetadata],
    artifact_path: Path,
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    total_payload_bytes = sum(item.nbytes for item in metadata)
    return {
        "schema_version": SELECTIVE_MATERIALIZATION_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "selector": LAYER_ROUTER_NORMS_SELECTOR,
        "layer": int(layer),
        "source": {
            "snapshot_dir": str(index.snapshot_dir),
            "weight_index": MODEL_INDEX_FILENAME,
        },
        "artifact": _relative_or_absolute(artifact_path, artifact_path.parent),
        "requested_keys": list(keys),
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": total_payload_bytes,
        },
        "tensors": [item.to_manifest_dict(index.snapshot_dir) for item in metadata],
    }


def _enforce_budget(metadata: Sequence[TensorMetadata], *, max_tensors: int, max_bytes: int) -> None:
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if len(metadata) > max_tensors:
        raise ValueError(f"Requested {len(metadata)} tensors exceeds tensor budget {max_tensors}")
    selected_bytes = sum(item.nbytes for item in metadata)
    if selected_bytes > max_bytes:
        raise ValueError(f"Requested {selected_bytes} bytes exceeds byte budget {max_bytes}")


def _safetensors_nbytes(dtype: str, shape: Sequence[int]) -> int:
    numel = math.prod(shape)
    if dtype in _SAFETENSORS_DTYPE_BYTES:
        return int(numel * _SAFETENSORS_DTYPE_BYTES[dtype])
    if dtype.startswith("F4"):
        return int((numel + 1) // 2)
    raise ValueError(f"Unsupported safetensors dtype for budget accounting: {dtype}")


def _group_locations_by_shard(locations: Sequence[TensorLocation]) -> dict[str, list[TensorLocation]]:
    grouped: dict[str, list[TensorLocation]] = {}
    for location in locations:
        grouped.setdefault(location.shard_name, []).append(location)
    return grouped


def _group_metadata_by_shard(metadata: Sequence[TensorMetadata]) -> dict[str, list[TensorMetadata]]:
    grouped: dict[str, list[TensorMetadata]] = {}
    for item in metadata:
        grouped.setdefault(item.shard_name, []).append(item)
    return grouped


def _raise_if_missing_shard(shard_path: Path) -> None:
    if not shard_path.is_file():
        raise FileNotFoundError(f"Safetensor shard referenced by index does not exist: {shard_path}")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
