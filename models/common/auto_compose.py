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
    try:
        return ttnn.to_torch(tensor, mesh_composer=composer)
    except Exception as e:
        logger.error(f"Failed to convert tensor to torch with mesh_composer: {e}")
        raise


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
        return _compose_nd_sharded(mesh_device, placements, dist_shape)


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
) -> ttnn.CppMeshToTensor:
    """
    Handle ND (N>=2) case.

    For replicated mesh dims, we use dim 0 as convention (the composed result
    will include all replicas concatenated, which is typically not desired but
    is how the C++ API works).
    """
    dims = []
    shape_override = []
    for i, p in enumerate(placements):
        if isinstance(p, ttnn.PlacementShard):
            dims.append(p.dim)
            shape_override.append(dist_shape[i])
        else:
            assert isinstance(p, ttnn.PlacementReplicate)
            # [INFO] steal from TensorDistribution2x4Test test case in test_distributed_tensor.cpp
            # Replicated: use dim 0 as convention
            dims.append(0)
            # Replicated: use shape 1 to skip concatenation
            shape_override.append(1)

    composer_cfg = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape_override))
    return ttnn.create_mesh_composer(device, composer_cfg)
