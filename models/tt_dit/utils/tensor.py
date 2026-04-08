# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    import torch


def typed_tensor(
    x: torch.Tensor,
    dtype: ttnn.DataType,
    device: ttnn.Device | None = None,
    mesh_axis=None,
    shard_dim=None,
    layout=ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """
    Replicates or shards a tensor based on the mesh_axis and shard_dim
    """
    assert (mesh_axis is None) == (shard_dim is None)
    mesh_mapper = None
    if mesh_axis is not None:
        mapper_dims = [None, None]
        mapper_dims[mesh_axis] = shard_dim
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=mesh_mapper,
    )


def bf16_tensor(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    return typed_tensor(x, ttnn.bfloat16, device, mesh_axis, shard_dim, layout)


def float32_tensor(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    return typed_tensor(x, ttnn.float32, device, mesh_axis, shard_dim, layout)


def bf16_tensor_host(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    assert (mesh_axis is None) == (shard_dim is None)
    mesh_mapper = None
    if mesh_axis is not None:
        mapper_dims = [None, None]
        mapper_dims[mesh_axis] = shard_dim
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )


def typed_tensor_2dshard(
    x: torch.Tensor, device: ttnn.Device, shard_mapping: dict[int, int], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
) -> ttnn.Tensor:
    assert len(shard_mapping) == 2
    assert all(0 <= k <= 1 and 0 <= v < len(x.shape) for k, v in shard_mapping.items())
    mapper_dims = [None, None]
    for k, v in shard_mapping.items():
        mapper_dims[k] = v
    mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)
    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=mesh_mapper,
    )


def bf16_tensor_2dshard(
    x: torch.Tensor,
    device: ttnn.Device,
    shard_mapping: dict[int, int],
    layout=ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    return typed_tensor_2dshard(x, device, shard_mapping, layout, dtype=ttnn.bfloat16)


def from_torch(
    x: torch.Tensor,
    /,
    *,
    device: ttnn.MeshDevice | None = None,
    layout: ttnn.Layout = ttnn.Layout.TILE,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    pad_value: float | None = None,
    mesh_axes: Sequence[int | None | EllipsisType] | None = None,
    on_host: bool = False,
) -> ttnn.Tensor:
    """Convert a torch.Tensor to a ttnn.Tensor with convenient mesh distribution."""
    if mesh_axes is not None:
        if device is None:
            msg = "device must be specified if mesh_axes is given"
            raise ValueError(msg)

        mesh_rank = len(list(device.shape))
        mesh_axes = canonicalize_tensor_mesh_axes(mesh_axes, tensor_rank=len(x.shape), mesh_rank=mesh_rank)

        placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
        placements = [ttnn.PlacementShard(p) if p is not None else ttnn.PlacementReplicate() for p in placements]
        mesh_mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))
    else:
        mesh_mapper = None

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=None if on_host else device,
        mesh_mapper=mesh_mapper,
        pad_value=pad_value,
    )


def to_torch(
    x: ttnn.Tensor,
    /,
    *,
    mesh_axes: Sequence[int | None | EllipsisType] | None = None,
    composer_device: ttnn.MeshDevice | None = None,
) -> torch.Tensor:
    """Converts a ttnn.Tensor to a torch.Tensor.

    Strips away redundant data returned by calling ttnn.to_torch on a replicated tensor. If the
    tensor is distributed and not on device, composer_device must be provided.
    """
    if x.tensor_topology().distribution_shape().mesh_size() == 1:
        return ttnn.to_torch(x)

    if mesh_axes is None:
        mesh_axes = (None,) * len(x.shape)

    composer_device = composer_device or x.device()
    if composer_device is None:
        msg = "composer_device must be specified for distributed host tensors"
        raise ValueError(msg)

    mesh_rank = len(list(composer_device.shape))
    mesh_axes = canonicalize_tensor_mesh_axes(mesh_axes, tensor_rank=len(x.shape), mesh_rank=mesh_rank)

    replicated_mesh_axes = list(set(range(mesh_rank)) - {axis for axis in mesh_axes if axis is not None})
    mesh_axes = replicated_mesh_axes + list(mesh_axes)

    placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
    assert all(p is not None for p in placements)

    mesh_composer = ttnn.create_mesh_composer(composer_device, ttnn.MeshComposerConfig(placements))

    x = x.reshape([1] * len(replicated_mesh_axes) + list(x.shape))
    return ttnn.to_torch(x, mesh_composer=mesh_composer)[(0,) * len(replicated_mesh_axes)]


def verify_tensor_mesh_axes(mesh_axes: Sequence[int | None], /, *, tensor_rank: int, mesh_rank: int) -> None:
    """Validates tensor mesh axes specification."""
    if len(mesh_axes) != tensor_rank:
        msg = f"mesh axis list {tuple(mesh_axes)} should have length {tensor_rank}"
        raise ValueError(msg)

    for axis in mesh_axes:
        if axis is not None and (axis < 0 or axis >= mesh_rank):
            msg = f"all mesh axes in mesh axis list {tuple(mesh_axes)} should be positive and smaller than {mesh_rank}"
            raise ValueError(msg)

    non_none_values = [p for p in mesh_axes if p is not None]
    if len(non_none_values) != len(set(non_none_values)):
        msg = f"mesh axis list {tuple(mesh_axes)} contains duplicate mesh axis assignments"
        raise ValueError(msg)


def canonicalize_tensor_mesh_axes(
    mesh_axes: Sequence[int | None | EllipsisType], /, *, tensor_rank: int, mesh_rank: int
) -> tuple[int | None, ...]:
    """Canonicalizes mesh axes specification by expanding Ellipsis and validating."""
    mesh_axes = list(mesh_axes)

    if Ellipsis in mesh_axes:
        if mesh_axes.count(Ellipsis) > 1:
            msg = "mesh_axes can contain at most one Ellipsis"
            raise ValueError(msg)

        ellipsis_index = mesh_axes.index(Ellipsis)
        mesh_axes[ellipsis_index : ellipsis_index + 1] = [None] * (tensor_rank - len(mesh_axes) + 1)

    verify_tensor_mesh_axes(mesh_axes, tensor_rank=tensor_rank, mesh_rank=mesh_rank)

    return tuple(mesh_axes)


def _invert_placements(placements: Sequence[int | None], *, output_rank: int) -> tuple[int | None, ...]:
    out = [None] * output_rank

    for i, p in enumerate(placements):
        if p is not None:
            out[p] = i

    return tuple(out)


def local_device_to_torch(tt_tensor: ttnn.Tensor) -> torch.Tensor:
    """Convert a ttnn device tensor to a torch tensor by reading from the local device.

    In a distributed environment, iterates over the mesh coordinates to find the
    tensor shard that belongs to the local device before calling ``ttnn.to_torch``.
    """
    mesh_device = tt_tensor.device()
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    coords = list(tt_tensor.tensor_topology().mesh_coords())
    device_tensors = ttnn.get_device_tensors(tt_tensor)

    torch_tensor = None
    for coord, device_tensor in zip(coords, device_tensors):
        if view is None or view.is_local(coord):
            torch_tensor = ttnn.to_torch(device_tensor)
            break

    if torch_tensor is None:
        msg = "Failed to find local device tensor"
        raise RuntimeError(msg)
    return torch_tensor


_to_torch_zero_copy_warned = False


def _to_torch_zero_copy(t: ttnn.Tensor) -> torch.Tensor:
    """Convert a host ttnn tensor to a PyTorch tensor, preferring zero-copy.

    Uses ``to_torch_with_padded_shape`` when available — for ROW_MAJOR host
    tensors this wraps the existing buffer directly (zero-copy) instead of
    copying through ``decode_tensor_data`` as ``to_torch`` always does.

    Falls back to ``to_torch`` if the method is removed, with a one-time
    warning so the performance regression is visible.

    TODO: Once ``to_torch`` supports a ``padded_output`` parameter (or the
    zero-copy path becomes the default for ROW_MAJOR), switch to that and
    remove this helper.
    """
    global _to_torch_zero_copy_warned
    try:
        return t.to_torch_with_padded_shape()
    except AttributeError:
        if not _to_torch_zero_copy_warned:
            import logging

            logging.getLogger(__name__).warning(
                "to_torch_with_padded_shape unavailable, falling back to to_torch (slower d2h)"
            )
            _to_torch_zero_copy_warned = True
        return ttnn.to_torch(t)


def fast_device_to_host(
    tt_tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    concat_dims: list[int | None],
    ccl_manager=None,
) -> torch.Tensor:
    """Fast D2H transfer using async DMA and zero-copy to_torch.

    On a single-host system, this avoids the on-device all_gather by reading
    all per-device shards concurrently with async DMA, converting to PyTorch
    with zero-copy when possible, and concatenating on host.

    On a multi-host (distributed) system, each host can only access its local
    devices, so this falls back to on-device all_gather via *ccl_manager*
    followed by a local-device read.

    Args:
        tt_tensor: Multi-device ttnn tensor on ``mesh_device``.
        mesh_device: The mesh device.
        concat_dims: Per mesh axis, the tensor dimension to concatenate along,
            or ``None`` to skip that axis.  E.g. ``[3, 4]`` means concatenate
            along dim 3 for mesh axis 0 and dim 4 for mesh axis 1.
        ccl_manager: Optional :class:`CCLManager` instance.  Required for
            multi-host environments where only local devices are accessible.
    """
    # Multi-host: can only access local devices, must all_gather on device first.
    if ttnn.using_distributed_env():
        if ccl_manager is None:
            msg = "fast_device_to_host requires ccl_manager in a distributed " "(multi-host) environment"
            raise ValueError(msg)
        return ccl_manager.device_to_host(tt_tensor, concat_dims)

    from concurrent.futures import ThreadPoolExecutor

    import torch

    mesh_shape = tuple(mesh_device.shape)

    # Get mesh coordinates before issuing DMA — topology is on the original
    # device tensor and maps each shard index to its (row, col) mesh position.
    mesh_coords = list(tt_tensor.tensor_topology().mesh_coords())
    device_tensors = ttnn.get_device_tensors(tt_tensor)

    # Async DMA: issue all transfers then sync once
    host_tensors = [dt.cpu(blocking=False) for dt in device_tensors]
    ttnn.synchronize_device(mesh_device)

    # Zero-copy to_torch when available, otherwise standard to_torch
    with ThreadPoolExecutor(max_workers=len(host_tensors)) as pool:
        shards = list(pool.map(_to_torch_zero_copy, host_tensors))

    # Trim physical (tile-padded) shape to logical shape — view, no copy
    logical_shape = list(host_tensors[0].shape)
    shards = [s[tuple(slice(0, d) for d in logical_shape)] for s in shards]

    # Build coord→shard mapping using explicit mesh coordinates rather than
    # assuming get_device_tensors() returns shards in row-major order.
    shards_by_coord = {(int(c[0]), int(c[1])): s for c, s in zip(mesh_coords, shards)}

    # Validate: if a mesh axis is not gathered (concat_dims[axis] is None),
    # the tensor must be replicated along that axis (size 1), otherwise we'd
    # silently drop shards.
    for axis in range(len(concat_dims)):
        if concat_dims[axis] is None and mesh_shape[axis] > 1:
            msg = (
                f"concat_dims[{axis}] is None (no gather) but mesh_shape[{axis}]={mesh_shape[axis]} > 1. "
                f"This would drop shards. Either gather along this axis or ensure the tensor is replicated."
            )
            raise ValueError(msg)

    # Reassemble from 2D mesh using explicit coordinate lookup.
    if concat_dims[0] is not None and concat_dims[1] is not None:
        rows = []
        for r in range(mesh_shape[0]):
            row_shards = [shards_by_coord[(r, c)] for c in range(mesh_shape[1])]
            rows.append(torch.cat(row_shards, dim=concat_dims[1]))
        return torch.cat(rows, dim=concat_dims[0])
    elif concat_dims[0] is not None:
        return torch.cat(
            [shards_by_coord[(r, 0)] for r in range(mesh_shape[0])],
            dim=concat_dims[0],
        )
    elif concat_dims[1] is not None:
        return torch.cat(
            [shards_by_coord[(0, c)] for c in range(mesh_shape[1])],
            dim=concat_dims[1],
        )
    else:
        return shards[0]


def upsample(
    x: ttnn.Tensor,
    /,
    *,
    scale_factor: int,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Wrapper around ttnn.upsample that allows for padded tensors."""
    change_layout = x.layout == ttnn.TILE_LAYOUT and x.shape != x.padded_shape

    if change_layout:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    x = ttnn.upsample(x, scale_factor=scale_factor, memory_config=memory_config)

    if change_layout:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    return x


def unflatten(x: ttnn.Tensor, dim: int, sizes: Sequence[int]) -> ttnn.Tensor:
    """Expands a dimension of the input tensor over multiple dimensions.

    Args:
        x: ttnn.Tensor
            The input tensor to unflatten
        dim: int
            Dimension to be unflattened, specified as an index into input.shape.
        sizes: Sequence[int]
            New shape of the unflattened dimension. One of its elements can be -1 in which case the corresponding output dimension is inferred. Otherwise, the product of sizes must equal
    Returns:
        ttnn.Tensor
            The unflattened tensor.
    """
    assert (
        x.shape[dim] % abs(math.prod(sizes)) == 0
    ), f"The total number of elements in the new shape {sizes} must be equal or a factor of the number of elements (when using inferred dimensions) in the original shape {x.shape[dim]}"
    new_shape = list(x.shape)
    if dim == -1:
        new_shape[-1:] = sizes
    else:
        new_shape[dim : dim + 1] = sizes
    return ttnn.reshape(x, new_shape)
