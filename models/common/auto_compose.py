# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Automatic composition of multi-device sharded tensors using TensorTopology.

This module provides utilities to infer the correct MeshToTensor composer from a
sharded ttnn.Tensor's topology metadata and use it to compose shards on host.
"""

from typing import Optional

import torch
from loguru import logger

import ttnn

# ======================================================================================
# Public API
# ======================================================================================


def to_torch_auto_compose(tensor: ttnn.Tensor, device: Optional[ttnn.MeshDevice] = None) -> torch.Tensor:
    """
    Convert a (possibly multi-device) TTNN tensor to torch, automatically
    composing shards based on the tensor's topology.

    Args:
        tensor: The distributed tensor to convert
        device: Optional MeshDevice to use when the tensor lives on host

    Returns:
        PyTorch tensor with shards composed
    """
    composer = _infer_mesh_composer_from_topology(tensor, device=device)
    if composer is not None:
        try:
            return ttnn.to_torch(tensor, mesh_composer=composer)
        except RuntimeError as e:
            # C++ compositor requires concat dims to be unique. Some topologies legitimately
            # need sequential concatenation along the same tensor dim (e.g. 2D sharding on W for both mesh axes).
            if "dims must be unique" not in str(e):
                raise
    return _manual_to_torch_compose(tensor, device=device)


def extract_tensor_topology_info(
    tensor: ttnn.Tensor,
) -> tuple[list[object], list[int]]:
    """
    Extract placements and distribution shape from a tensor's topology.

    Returns:
        (placements, dist_shape)
    """
    topology = tensor.tensor_topology()
    placements = topology.placements()
    dist_shape = list(topology.distribution_shape())
    return placements, dist_shape


def get_device_from_tensor(tensor: ttnn.Tensor) -> Optional[ttnn.MeshDevice]:
    """Get device from tensor or fallback to provided mesh_device."""
    device = tensor.device()
    # tensor.device() returns None if the tensor is on the host (ttnn/core/tensor/tensor.cpp --> Tensor::device())
    if device is None:
        logger.debug("tensor.device() returns None, tensor is on the host")
    else:
        logger.debug(f"tensor.device() returns {device}")

    return device


# ======================================================================================
# Private Implementation
# ======================================================================================


def _infer_mesh_composer_from_topology(
    tensor: ttnn.Tensor, *, device: Optional[ttnn.MeshDevice] = None
) -> Optional[ttnn.CppMeshToTensor]:
    """
    Return a MeshToTensor composer inferred from the tensor's TensorTopology,
    or None if no composition is needed (fully replicated, single-device).

    Note: For ND meshes with replicated dimensions, the composer will concatenate
    all replicas, resulting in duplicated data. Callers may want to slice the
    result if only one copy is desired.

    Args:
        tensor: The distributed tensor to infer composer for

    Returns:
        MeshToTensor composer or None if no composition needed
    """
    placements, dist_shape = extract_tensor_topology_info(tensor)

    # No distribution or trivial 1-device case
    if len(dist_shape) == 0 or (len(dist_shape) == 1 and dist_shape[0] == 1):
        return None

    tensor_device = get_device_from_tensor(tensor)
    mesh_device = tensor_device or device
    if mesh_device is None:
        # As a last resort, try default device for backward-compatibility
        mesh_device = ttnn.GetDefaultDevice()
        if mesh_device is None:
            raise RuntimeError(
                "Tensor is on host and no mesh_device provided. "
                "Pass device=... to to_torch_auto_compose or set a default via ttnn.SetDefaultDevice(...)."
            )

    # Must match length (should be guaranteed by C++ TT_FATAL in ttnn/core/distributed/distributed_tensor.cpp)
    assert len(dist_shape) == len(placements)

    if len(dist_shape) == 1 and mesh_device.shape.dims() == 1:
        return _compose_1d_sharded(mesh_device, placements, dist_shape)
    else:
        # N >= 2 dimensions
        return _compose_nd_sharded(mesh_device, placements, dist_shape, tensor_rank=len(tensor.shape))


def _compose_1d_sharded(
    device: ttnn.MeshDevice,
    placements: list[object],
    dist_shape: list[int],
) -> Optional[ttnn.CppMeshToTensor]:
    """Handle 1D case - returns None if fully replicated."""
    p = placements[0]
    if isinstance(p, ttnn.PlacementShard):
        # Use ND composer with shape override to match the tensor's distribution
        composer_cfg = ttnn.MeshComposerConfig(dims=[p.dim], mesh_shape_override=ttnn.MeshShape(dist_shape))
        return ttnn.create_mesh_composer(device, composer_cfg)
    # Fully replicated - no composition needed
    return None


def _compose_nd_sharded(
    device: ttnn.MeshDevice,
    placements: list[object],
    dist_shape: list[int],
    tensor_rank: int,
) -> Optional[ttnn.CppMeshToTensor]:
    """
    Handle ND (N>=2) case.

    For replicated mesh dims, we use dim 0 as convention (the composed result
    will include all replicas concatenated, which is typically not desired but
    is how the C++ API works).
    """
    dims: list[int] = []
    shape_override: list[int] = []
    used_dims: set[int] = set()
    for i, p in enumerate(placements):
        if isinstance(p, ttnn.PlacementShard):
            dims.append(p.dim)
            shape_override.append(dist_shape[i])
            used_dims.add(p.dim)
        else:
            assert isinstance(p, ttnn.PlacementReplicate)
            # Replicated mesh dimension: we don't want concatenation (shape_override=1),
            # but C++ requires `dims` to be unique.
            replica_dim = next((d for d in range(tensor_rank) if d not in used_dims), 0)
            dims.append(replica_dim)
            used_dims.add(replica_dim)
            # Replicated: use shape 1 to skip concatenation
            shape_override.append(1)

    # If concat dims are not unique, C++ MeshToTensor can't compose in one pass.
    if len(set(dims)) != len(dims):
        return None

    composer_cfg = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape_override))
    return ttnn.create_mesh_composer(device, composer_cfg)


def _manual_to_torch_compose(tensor: ttnn.Tensor, *, device: Optional[ttnn.MeshDevice]) -> torch.Tensor:
    """
    Python fallback composer for distributed tensors.

    This is slower than the C++ MeshToTensor path but can handle cases that require
    sequential concatenation along the same tensor dim.
    """
    placements, dist_shape = extract_tensor_topology_info(tensor)

    tensor_device = get_device_from_tensor(tensor)
    mesh_device = tensor_device or device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise RuntimeError(
            "Tensor is on host and no mesh_device provided. "
            "Pass device=... to to_torch_auto_compose or set a default via ttnn.SetDefaultDevice(...)."
        )

    mesh_shape = list(mesh_device.shape)
    device_tensors = ttnn.get_device_tensors(tensor)

    def _coords_row_major(shape: list[int]) -> list[tuple[int, ...]]:
        coords: list[tuple[int, ...]] = [()]
        for s in shape:
            coords = [c + (i,) for c in coords for i in range(s)]
        return coords

    def _flatten(coord: tuple[int, ...], shape: list[int]) -> int:
        idx = 0
        stride = 1
        for c, s in zip(reversed(coord), reversed(shape)):
            idx += c * stride
            stride *= s
        return idx

    # Map mesh_coord -> torch shard (assumes get_device_tensors order is row-major over mesh coords).
    mesh_coord_to_torch: dict[tuple[int, ...], torch.Tensor] = {}
    for coord in _coords_row_major(mesh_shape):
        mesh_coord_to_torch[coord] = ttnn.to_torch(device_tensors[_flatten(coord, mesh_shape)])

    # Map distribution coords -> mesh coords, then build dist_coord -> torch shard.
    mapping = ttnn.compute_distribution_to_mesh_mapping(ttnn.MeshShape(dist_shape), mesh_device.shape)
    shards: dict[tuple[int, ...], torch.Tensor] = {}
    for flat_i, mesh_coord in enumerate(mapping):
        # unravel flat_i into dist_coord (row-major)
        rem = flat_i
        dist_coord_rev: list[int] = []
        for s in reversed(dist_shape):
            dist_coord_rev.append(rem % s)
            rem //= s
        dist_coord = tuple(reversed(dist_coord_rev))
        mesh_coord_tuple = tuple(list(mesh_coord))
        shards[dist_coord] = mesh_coord_to_torch[mesh_coord_tuple]

    # Heuristic: some ops (e.g. all-reduce) can leave topology metadata in a state that implies
    # sharding on an axis where data is actually replicated. This can happen in practice when
    # the reduced axis still reports PlacementShard. If shards are identical across an axis,
    # treat that axis as replicated for host composition.
    effective_placements: list[object] = list(placements)
    sample_elems = 256
    base = tuple(0 for _ in dist_shape)
    for axis, p in enumerate(placements):
        if not isinstance(p, ttnn.PlacementShard):
            continue
        if dist_shape[axis] <= 1:
            continue
        coord0 = base[:axis] + (0,) + base[axis + 1 :]
        coord1 = base[:axis] + (1,) + base[axis + 1 :]
        t0 = shards[coord0].reshape(-1)[:sample_elems]
        t1 = shards[coord1].reshape(-1)[:sample_elems]
        if torch.equal(t0, t1):
            effective_placements[axis] = ttnn.PlacementReplicate()

    # Compose across distribution axes right-to-left.
    for axis in range(len(dist_shape) - 1, -1, -1):
        p = effective_placements[axis]
        new_shards: dict[tuple[int, ...], torch.Tensor] = {}
        if isinstance(p, ttnn.PlacementReplicate):
            # Pick the first replica along this axis (matches C++ shape_override=1 convention).
            for coord, val in shards.items():
                if coord[axis] != 0:
                    continue
                new_coord = coord[:axis] + coord[axis + 1 :]
                new_shards[new_coord] = val
        else:
            assert isinstance(p, ttnn.PlacementShard)
            for base in {c[:axis] + c[axis + 1 :] for c in shards.keys()}:
                pieces = [shards[base[:axis] + (i,) + base[axis:]] for i in range(dist_shape[axis])]
                new_shards[base] = torch.cat(pieces, dim=p.dim)
        shards = new_shards

    assert len(shards) == 1
    return next(iter(shards.values()))
