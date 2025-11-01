#!/usr/bin/env python3
"""
Distribute a torch.Tensor over a mesh using the same topology as a reference TTNN tensor.

This mirrors the composition logic in `auto_compose.py` but in reverse: we infer a
TensorToMesh mapper from the reference tensor's TensorTopology and use it to distribute
the torch tensor accordingly.
"""

from typing import Optional

import torch

import ttnn

from .auto_compose import extract_tensor_topology_info, get_device_from_tensor


def from_torch_dist_as(
    from_tensor_pt: torch.Tensor, as_tensor_tt: ttnn.Tensor, device: Optional[ttnn.MeshDevice] = None
) -> ttnn.Tensor:
    """
    Distribute a torch.Tensor over a mesh using the same topology as an existing TTNN tensor.

    Args:
        tensor_pt: Source PyTorch tensor on host
        tensor_tt: Reference TTNN tensor whose topology (placements + distribution shape) will be mirrored

    Returns:
        A TTNN tensor distributed according to `tensor_tt`'s topology.
    """
    mapper, device = _infer_mesh_mapper_from_topology(as_tensor_tt, device=device)

    # Usage Patterns: unlike ttnn.to_torch, `device` is required here!
    # Pattern 1: Using mesh_mapper without device (tensor stays in host memory) Programming_Mesh_of_Devices_with_TT-NN.md:370-375
    # Then transfer to device separately: Programming_Mesh_of_Devices_with_TT-NN.md:404-405
    # Pattern 2: Using both mesh_mapper and device together (direct to device) llms.md:1204-1218
    return ttnn.from_torch(
        from_tensor_pt,
        dtype=getattr(as_tensor_tt, "dtype", None),
        layout=getattr(as_tensor_tt, "layout", None),
        device=device,
        mesh_mapper=mapper,
    )


def _infer_mesh_mapper_from_topology(
    tensor: ttnn.Tensor, *, device: Optional[ttnn.MeshDevice] = None
) -> Optional[ttnn.CppTensorToMesh]:
    """
    Return a TensorToMesh mapper inferred from the tensor's TensorTopology,
    or (None, mesh_device) if no distribution is needed (fully replicated, single-device).
    """
    placements, dist_shape = extract_tensor_topology_info(tensor)

    # No distribution or trivial 1-device case
    if len(dist_shape) == 0 or (len(dist_shape) == 1 and dist_shape[0] == 1):
        mesh_device = get_device_from_tensor(tensor) or device or ttnn.GetDefaultDevice()
        if mesh_device is None:
            raise RuntimeError(
                "Tensor is on host and no mesh_device provided. " "Set a default via ttnn.SetDefaultDevice(...)."
            )
        return None, mesh_device

    tensor_device = get_device_from_tensor(tensor)
    mesh_device = tensor_device or device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
        if mesh_device is None:
            raise RuntimeError(
                "Tensor is on host and no mesh_device provided. " "Set a default via ttnn.SetDefaultDevice(...)."
            )

    assert len(dist_shape) == len(placements)

    if len(dist_shape) == 1:
        return _map_1d(mesh_device, placements, dist_shape), mesh_device
    else:
        return _map_nd(mesh_device, placements, dist_shape), mesh_device


def _map_1d(
    device: ttnn.MeshDevice,
    placements: list[object],
    dist_shape: list[int],
) -> Optional[ttnn.CppTensorToMesh]:
    """
    Build a 1D TensorToMesh mapper. Returns None if fully trivial (handled earlier).
    """
    p = placements[0]
    if isinstance(p, ttnn.PlacementShard):
        mapper_cfg = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(p.dim)],
            mesh_shape_override=ttnn.MeshShape(dist_shape),
        )
        return ttnn.create_mesh_mapper(device, mapper_cfg)
    else:
        # Replicate across the 1D mesh extent
        mapper_cfg = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape(dist_shape),
        )
        return ttnn.create_mesh_mapper(device, mapper_cfg)


def _map_nd(
    device: ttnn.MeshDevice,
    placements: list[object],
    dist_shape: list[int],
) -> ttnn.CppTensorToMesh:
    """
    Build an ND TensorToMesh mapper that mirrors the tensor's placements and distribution shape.
    """
    mapper_placements = []
    for p in placements:
        if isinstance(p, ttnn.PlacementShard):
            mapper_placements.append(ttnn.PlacementShard(p.dim))
        else:
            assert isinstance(p, ttnn.PlacementReplicate)
            mapper_placements.append(ttnn.PlacementReplicate())

    mapper_cfg = ttnn.MeshMapperConfig(placements=mapper_placements, mesh_shape_override=ttnn.MeshShape(dist_shape))
    return ttnn.create_mesh_mapper(device, mapper_cfg)
