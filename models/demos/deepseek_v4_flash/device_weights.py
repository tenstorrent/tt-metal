# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from models.demos.deepseek_v4_flash.weight_inventory import (
    WeightPlacement,
    WeightTensorRecord,
    plan_weight_placements,
    read_weight_tensor_records,
)


class PreprocessedWeightIndex:
    """Index canonical TT-preprocessed tensors by key without loading payloads."""

    def __init__(self, preprocessed_dir: str | Path):
        self.preprocessed_dir = Path(preprocessed_dir).expanduser().resolve()
        records = read_weight_tensor_records(self.preprocessed_dir)
        self.records_by_key = {record.key: record for record in records}
        if len(self.records_by_key) != len(records):
            raise ValueError(f"Duplicate tensor keys found in {self.preprocessed_dir}")

    def record(self, key: str) -> WeightTensorRecord:
        try:
            return self.records_by_key[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is not present in {self.preprocessed_dir}") from exc

    def load_torch(self, key: str) -> torch.Tensor:
        record = self.record(key)
        with safe_open(self.preprocessed_dir / record.artifact, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).contiguous()

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.records_by_key))


class TtDeviceWeightOwner:
    """Own TTNN tensors loaded from a TT-preprocessed DeepSeek V4 Flash checkpoint.

    This class is intentionally a low-level owner, not a model executor. It
    centralizes placement, submesh lifetime, and TTNN tensor ownership so the
    integrated runtime does not keep scattering ad hoc ``from_torch`` calls
    through smoke modules.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        *,
        mesh_device,
        mesh_shape: tuple[int, int] = (2, 4),
        dtype=None,
        layout=None,
        memory_config=None,
    ):
        self.index = PreprocessedWeightIndex(preprocessed_dir)
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(int(dim) for dim in mesh_shape)
        self.dtype = dtype
        self.layout = layout
        self.memory_config = memory_config
        self._placements_by_key = {
            placement.key: placement
            for placement in plan_weight_placements(tuple(self.index.records_by_key.values()), mesh_shape=self.mesh_shape)
        }
        self.tensors: dict[str, Any] = {}
        self._submeshes: dict[tuple[int, int], object] = {}

    @property
    def owned_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.tensors))

    def placement(self, key: str) -> WeightPlacement:
        try:
            return self._placements_by_key[key]
        except KeyError as exc:
            self.index.record(key)
            raise KeyError(f"No placement exists for tensor {key!r}") from exc

    def load_keys(self, keys: Iterable[str]) -> Mapping[str, Any]:
        for key in keys:
            if key not in self.tensors:
                self.tensors[key] = self._load_one(key)
        return {key: self.tensors[key] for key in keys}

    def get(self, key: str):
        if key not in self.tensors:
            raise KeyError(f"Tensor {key!r} has not been loaded; call load_keys first")
        return self.tensors[key]

    def close(self) -> None:
        if not self._submeshes:
            return
        import ttnn

        for submesh in reversed(tuple(self._submeshes.values())):
            ttnn.close_mesh_device(submesh)
        self._submeshes.clear()

    def _load_one(self, key: str):
        import ttnn

        record = self.index.record(key)
        placement = self.placement(key)
        tensor = self.index.load_torch(key)
        dtype = self.dtype if self.dtype is not None else _default_ttnn_dtype(ttnn, record, tensor)
        layout = self.layout if self.layout is not None else _default_layout(ttnn, tensor)
        memory_config = self.memory_config if self.memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

        if placement.strategy == "expert_home_device":
            device = self._submesh_for(placement.devices[0])
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        elif placement.strategy in ("tp_shard_replicate_ep", "replicate_all"):
            device = self.mesh_device
            mesh_mapper = _mesh_mapper_for_placement(ttnn, self.mesh_device, placement, tensor)
        else:
            raise ValueError(f"Tensor {key!r} uses non-device placement strategy {placement.strategy!r}")

        return ttnn.as_tensor(
            tensor,
            dtype=dtype,
            device=device,
            layout=layout,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _submesh_for(self, coord: tuple[int, int]):
        if coord not in self._submeshes:
            import ttnn

            self._submeshes[coord] = self.mesh_device.create_submesh(
                ttnn.MeshShape(1, 1),
                offset=ttnn.MeshCoordinate(*coord),
            )
        return self._submeshes[coord]


def _default_ttnn_dtype(ttnn, record: WeightTensorRecord, tensor: torch.Tensor):
    if record.dtype in ("I32", "U32", "I64", "U64"):
        return None
    if tensor.dtype in (torch.uint8, torch.int8):
        return None
    return ttnn.bfloat16


def _default_layout(ttnn, tensor: torch.Tensor):
    return ttnn.TILE_LAYOUT if tensor.ndim >= 2 else ttnn.ROW_MAJOR_LAYOUT


def _mesh_mapper_for_placement(ttnn, mesh_device, placement: WeightPlacement, tensor: torch.Tensor):
    if placement.strategy == "tp_shard_replicate_ep" and tensor.ndim >= 2:
        return ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, -1), mesh_shape=list(mesh_device.shape))
    return ttnn.ReplicateTensorToMesh(mesh_device)
