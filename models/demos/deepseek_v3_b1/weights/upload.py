# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import ttnn
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlappedTensor


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


def extract_backing_tensors(*weight_structs: Any) -> list[ttnn.Tensor]:
    """Return unique backing tensors from nested weight dataclasses."""
    out: list[ttnn.Tensor] = []
    seen_ids: set[tuple[str, int]] = set()

    def _add_tensor(tensor: ttnn.Tensor) -> None:
        tensor_key = tensor_identity_key(tensor)
        if tensor_key not in seen_ids:
            seen_ids.add(tensor_key)
            out.append(tensor)

    def _walk(value: Any) -> None:
        if isinstance(value, OverlappedTensor):
            _add_tensor(value.fused_tensor)
            return
        if isinstance(value, ttnn.Tensor):
            _add_tensor(value)
            return
        if is_dataclass(value):
            for field in fields(value):
                _walk(getattr(value, field.name))
            return
        if isinstance(value, list | tuple):
            for item in value:
                _walk(item)

    for struct in weight_structs:
        _walk(struct)

    return out


def two_phase_upload(device, host_tensors: list[ttnn.Tensor]) -> list[ttnn.Tensor]:
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


def rebuild_with_device_tensors(host_struct: Any, host_to_device: dict[tuple[str, int], ttnn.Tensor]) -> Any:
    """Rebuild nested weight dataclasses by replacing host tensors with device tensors."""

    def _replace(value: Any) -> Any:
        if isinstance(value, OverlappedTensor):
            fused_tensor = host_to_device[tensor_identity_key(value.fused_tensor)]
            return OverlappedTensor(
                fused_tensor=fused_tensor,
                tensor_shape=value.tensor_shape,
                shard_shape=value.shard_shape,
                core_range_set=value.core_range_set,
                dtype=value.dtype,
                tile_shape=value.tile_shape,
                byte_offset=value.byte_offset,
                total_size=value.total_size,
            )
        if isinstance(value, ttnn.Tensor):
            return host_to_device[tensor_identity_key(value)]
        if is_dataclass(value):
            rebuilt_fields = {field.name: _replace(getattr(value, field.name)) for field in fields(value)}
            return type(value)(**rebuilt_fields)
        if isinstance(value, list):
            return [_replace(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_replace(item) for item in value)
        return value

    return _replace(host_struct)
