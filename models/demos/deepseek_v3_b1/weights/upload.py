# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from types import UnionType
from typing import Protocol, Union, get_args, get_origin, get_type_hints

import ttnn
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlappedTensor

TensorKey = tuple[str, int]
TensorMap = dict[TensorKey, ttnn.Tensor]
WeightTensor = ttnn.Tensor | OverlappedTensor


class Uploadable(Protocol):
    def backing_tensors(self, *, skip_fields: tuple[str, ...] | set[str] | None = None) -> list[ttnn.Tensor]:
        ...

    def with_device_tensors(self, tensor_map: TensorMap, *, allow_unmapped: bool = False):
        ...


def _tensor_is_on_device(t: ttnn.Tensor) -> bool:
    return ttnn.is_tensor_storage_on_device(t)


class UploadableMixin:
    """Typed helper for extracting and rebuilding weight dataclasses.

    Tensors that are already on device are skipped by ``backing_tensors``
    (they don't need to be uploaded again) and passed through unchanged
    by ``with_device_tensors`` regardless of whether they appear in the
    rebuild ``tensor_map``.  This lets callers stage some tensors to the
    device early (e.g. attn/shared L1 lockstep weights for the SRAM hot
    expert trim) and still hand the same dataclass to ``two_phase_upload``
    later for the remaining host-staged tensors.
    """

    def backing_tensors(self, *, skip_fields: tuple[str, ...] | set[str] | None = None) -> list[ttnn.Tensor]:
        skip = set(skip_fields) if skip_fields else set()
        out: list[ttnn.Tensor] = []
        seen_ids: set[TensorKey] = set()
        hints = get_type_hints(type(self))
        for field in fields(self):
            if field.name in skip:
                continue
            annotation = hints[field.name]
            value = getattr(self, field.name)
            kind, allows_none = _classify_annotation(annotation)
            if value is None:
                if allows_none:
                    continue
                raise TypeError(f"Field {type(self).__name__}.{field.name} is None but annotation is non-optional")
            if kind == "tensor":
                if _tensor_is_on_device(value):
                    continue
                _append_unique_tensor(out, seen_ids, value, field.name)
            elif kind == "overlapped":
                if _tensor_is_on_device(value.fused_tensor):
                    continue
                _append_unique_tensor(out, seen_ids, value.fused_tensor, field.name)
            elif kind == "tensor_list":
                if not isinstance(value, list):
                    raise TypeError(
                        f"Field {type(self).__name__}.{field.name} must be list[ttnn.Tensor], got {type(value)}"
                    )
                for tensor in value:
                    if _tensor_is_on_device(tensor):
                        continue
                    _append_unique_tensor(out, seen_ids, tensor, field.name)
            elif kind == "passthrough":
                continue
            else:
                raise TypeError(f"Unsupported field annotation in {type(self).__name__}.{field.name}: {annotation}")
        return out

    def with_device_tensors(self, tensor_map: TensorMap, *, allow_unmapped: bool = False):
        hints = get_type_hints(type(self))
        rebuilt_fields: dict[str, object] = {}
        for field in fields(self):
            annotation = hints[field.name]
            value = getattr(self, field.name)
            kind, allows_none = _classify_annotation(annotation)
            if value is None:
                if allows_none:
                    rebuilt_fields[field.name] = None
                    continue
                raise TypeError(f"Field {type(self).__name__}.{field.name} is None but annotation is non-optional")
            if kind == "tensor":
                rebuilt_fields[field.name] = _resolve_tensor(
                    value, tensor_map, allow_unmapped, type(self).__name__, field.name
                )
            elif kind == "overlapped":
                if _tensor_is_on_device(value.fused_tensor):
                    rebuilt_fields[field.name] = value
                    continue
                key = tensor_identity_key(value.fused_tensor)
                if key in tensor_map:
                    fused_tensor = tensor_map[key]
                elif allow_unmapped:
                    rebuilt_fields[field.name] = value
                    continue
                else:
                    raise KeyError(f"Field {type(self).__name__}.{field.name}: fused tensor not found in upload map")
                rebuilt_fields[field.name] = OverlappedTensor(
                    fused_tensor=fused_tensor,
                    tensor_shape=value.tensor_shape,
                    shard_shape=value.shard_shape,
                    core_range_set=value.core_range_set,
                    dtype=value.dtype,
                    tile_shape=value.tile_shape,
                    byte_offset=value.byte_offset,
                    total_size=value.total_size,
                )
            elif kind == "tensor_list":
                if not isinstance(value, list):
                    raise TypeError(
                        f"Field {type(self).__name__}.{field.name} must be list[ttnn.Tensor], got {type(value)}"
                    )
                rebuilt_fields[field.name] = [
                    _resolve_tensor(t, tensor_map, allow_unmapped, type(self).__name__, field.name) for t in value
                ]
            elif kind == "passthrough":
                rebuilt_fields[field.name] = value
            else:
                raise TypeError(f"Unsupported field annotation in {type(self).__name__}.{field.name}: {annotation}")
        return type(self)(**rebuilt_fields)


def _resolve_tensor(
    value: ttnn.Tensor,
    tensor_map: TensorMap,
    allow_unmapped: bool,
    type_name: str,
    field_name: str,
) -> ttnn.Tensor:
    if _tensor_is_on_device(value):
        return value
    key = tensor_identity_key(value)
    if key in tensor_map:
        return tensor_map[key]
    if allow_unmapped:
        return value
    raise KeyError(f"Field {type_name}.{field_name}: tensor not found in upload map")


def _classify_annotation(annotation: object) -> tuple[str, bool]:
    origin = get_origin(annotation)
    if origin is None:
        if annotation is ttnn.Tensor:
            return "tensor", False
        if annotation is OverlappedTensor:
            return "overlapped", False
        if isinstance(annotation, type) and getattr(annotation, "__upload_passthrough__", False):
            return "passthrough", False
        raise TypeError(f"Unsupported field annotation: {annotation}")

    if origin is list:
        args = get_args(annotation)
        if len(args) == 1 and args[0] is ttnn.Tensor:
            return "tensor_list", False
        raise TypeError(f"Unsupported list field annotation: {annotation}")

    if origin is tuple:
        raise TypeError(f"tuple fields are not supported in uploadable weight dataclasses: {annotation}")

    if origin in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        allows_none = len(args) != len(get_args(annotation))
        if len(args) != 1:
            raise TypeError(f"Unsupported union field annotation: {annotation}")
        kind, _ = _classify_annotation(args[0])
        return kind, allows_none

    raise TypeError(f"Unsupported field annotation: {annotation}")


def _append_unique_tensor(out: list[ttnn.Tensor], seen_ids: set[TensorKey], tensor: object, field_name: str) -> None:
    if not isinstance(tensor, ttnn.Tensor):
        raise TypeError(f"Field {field_name} expected ttnn.Tensor, got {type(tensor)}")
    tensor_key = tensor_identity_key(tensor)
    if tensor_key not in seen_ids:
        seen_ids.add(tensor_key)
        out.append(tensor)


def tensor_identity_key(tensor: ttnn.Tensor) -> tuple[str, int]:
    """Return a stable tensor identity key across python wrapper instances."""
    tensor_id = getattr(tensor, "tensor_id", None)
    if tensor_id is not None:
        return ("tensor_id", int(tensor_id))
    return ("py_id", id(tensor))


def get_fd_grid(device) -> ttnn.CoreRangeSet:
    """Return the FD grid used for fast host-to-device transfers."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x <= 1 or compute_grid.y <= 0:
        return ttnn.CoreRangeSet([])
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid.x - 2, compute_grid.y - 1),
            )
        ]
    )


def split_core_ranges(
    tensor_grid: ttnn.CoreRangeSet,
    fd_grid: ttnn.CoreRangeSet,
) -> tuple[ttnn.CoreRangeSet, ttnn.CoreRangeSet]:
    """Split tensor shard grid into fast-dispatch and slow-dispatch subsets."""
    sd_filter = tensor_grid.subtract(fd_grid)
    fd_filter = tensor_grid.subtract(sd_filter)
    return fd_filter, sd_filter


def _upload_tensors(device: ttnn.MeshDevice, host_tensors: list[ttnn.Tensor]) -> list[ttnn.Tensor]:
    """Upload host tensors by writing FD shards first, then SD-only shards."""
    fd_grid = get_fd_grid(device)
    full_fd_jobs: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
    fd_partial_jobs: list[tuple[ttnn.Tensor, ttnn.Tensor, ttnn.CoreRangeSet]] = []
    sd_partial_jobs: list[tuple[ttnn.Tensor, ttnn.Tensor, ttnn.CoreRangeSet]] = []
    uploaded: list[ttnn.Tensor] = []

    for host_tensor in host_tensors:
        device_tensor = ttnn.allocate_tensor_on_device(host_tensor.spec, device)
        uploaded.append(device_tensor)

        if not host_tensor.is_sharded():
            full_fd_jobs.append((host_tensor, device_tensor))
            continue

        shard_spec = host_tensor.memory_config().shard_spec
        if shard_spec is None:
            full_fd_jobs.append((host_tensor, device_tensor))
            continue

        fd_filter, sd_filter = split_core_ranges(shard_spec.grid, fd_grid)
        if not fd_filter.empty():
            if sd_filter.empty():
                full_fd_jobs.append((host_tensor, device_tensor))
            else:
                fd_partial_jobs.append((host_tensor, device_tensor, fd_filter))
        if not sd_filter.empty():
            sd_partial_jobs.append((host_tensor, device_tensor, sd_filter))

    with ttnn.device.setup_fast_dispatch(device):
        for host_tensor, device_tensor in full_fd_jobs:
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        for host_tensor, device_tensor, core_filter in fd_partial_jobs:
            ttnn.copy_host_to_device_tensor_partial(host_tensor, device_tensor, core_filter)

    for host_tensor, device_tensor, core_filter in sd_partial_jobs:
        ttnn.copy_host_to_device_tensor_partial(host_tensor, device_tensor, core_filter)

    return uploaded


def two_phase_upload(device: ttnn.MeshDevice, host_weights: Uploadable):
    """Upload an Uploadable host weight struct and return device-backed copy.

    Tensors that are already on device (e.g. ones uploaded earlier by
    :func:`eager_upload_l1_lockstep`) are passed through unchanged.
    """
    host_tensors = host_weights.backing_tensors()
    device_tensors = _upload_tensors(device, host_tensors)
    host_to_device: TensorMap = {tensor_identity_key(host): dev for host, dev in zip(host_tensors, device_tensors)}
    return host_weights.with_device_tensors(host_to_device)


def eager_upload_l1_lockstep(
    device: ttnn.MeshDevice,
    host_weights: Uploadable,
    *,
    skip_fields: tuple[str, ...] | set[str] | None = None,
):
    """Upload only the L1-lockstep host tensors and return ``host_weights``
    with those fields swapped to device-resident tensors.

    Used to give the SRAM hot expert trim real allocator addresses for the
    attention L1 weights *before* it runs, so the trim sees the "attn first,
    SRAM below" allocation order its per-core invariant
    ``attn[c] + sram[c] <= cap`` relies on.

    ``skip_fields`` lets the caller exclude specific dataclass fields from
    the eager upload -- typically the shared expert weights, which live on
    the same SRAM cores but should *not* count against the SRAM cap (they
    get uploaded later by :func:`two_phase_upload` and land below the SRAM
    band in the ``worker_l1_size - cap`` reserve).

    DRAM tensors and per-core L1 passthrough fields (e.g. SRAM compressed
    expert slots) are left untouched and stay host-staged for the caller's
    later :func:`two_phase_upload` call.  Already-on-device tensors are
    skipped silently.
    """
    host_tensors = host_weights.backing_tensors(skip_fields=skip_fields)
    l1_lockstep = [t for t in host_tensors if t.memory_config().buffer_type == ttnn.BufferType.L1]
    if not l1_lockstep:
        return host_weights
    device_tensors = _upload_tensors(device, l1_lockstep)
    tensor_map: TensorMap = {tensor_identity_key(host): dev for host, dev in zip(l1_lockstep, device_tensors)}
    return host_weights.with_device_tensors(tensor_map, allow_unmapped=True)
