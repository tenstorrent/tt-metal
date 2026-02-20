# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for handling mesh devices and tensor placements in sweep tests.
Supports creating tensors on mesh devices with proper placement (Shard/Replicate).

Environment Variables:
    MESH_DEVICE_SHAPE: Mesh shape to use (e.g., "1x2", "2x4", "4x8")
                       If not set, uses single device (default)
                       If set, creates mesh device with that shape
                       Tests will fail naturally if mesh shape exceeds hardware
"""

import os
import torch
import ttnn
from typing import Optional, Dict, Tuple
import ast


def parse_placement_from_traced(tensor_placement: Optional[Dict]) -> Optional[ttnn.TensorMemoryLayout]:
    """
    Parse tensor placement from traced config and return appropriate mesh mapper.

    Args:
        tensor_placement: Dict with 'placement', 'distribution_shape', 'mesh_device_shape'
                         e.g., {'placement': "['PlacementShard(2)', 'PlacementShard(3)']", ...}

    Returns:
        Mesh mapper object (ShardTensor2dMesh or ReplicateTensorToMesh) or None
    """
    if not tensor_placement:
        return None

    try:
        placement_str = tensor_placement.get("placement", "")

        # Check if it's a replicate placement
        if "PlacementReplicate" in placement_str:
            return ttnn.ReplicateTensorToMesh

        # Check if it's a shard placement
        if "PlacementShard" in placement_str:
            # Extract shard dimensions
            # e.g., "['PlacementShard(2)', 'PlacementShard(3)']" -> shard on dims 2,3
            import re

            shard_dims = re.findall(r"PlacementShard\((\d+)\)", placement_str)

            if shard_dims:
                # For 2D mesh, we typically shard on the last dimension(s)
                # Return a shard mapper - the specific implementation depends on the operation
                mesh_shape_str = tensor_placement.get("mesh_device_shape", "[1, 1]")
                mesh_shape = ast.literal_eval(mesh_shape_str) if isinstance(mesh_shape_str, str) else mesh_shape_str

                # For now, return ShardTensor2dMesh which will shard based on mesh shape
                return ttnn.ShardTensor2dMesh(
                    mesh_device=None,  # Will be set later
                    dim=int(shard_dims[-1]) if shard_dims else -1,
                    mesh_shape=ttnn.MeshShape(*mesh_shape) if len(mesh_shape) == 2 else None,
                )
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse tensor placement: {e}")
        return None

    return None


def get_mesh_shape_from_machine_info(machine_info: Optional[Dict]) -> Optional[Tuple[int, int]]:
    """
    Extract mesh device shape from traced machine_info.

    Args:
        machine_info: Dict with 'mesh_device_shape', 'device_count', etc.

    Returns:
        Tuple of (rows, cols) or None if no mesh info
    """
    if not machine_info:
        return None

    mesh_shape = machine_info.get("mesh_device_shape")
    if not mesh_shape:
        return None

    # Handle both list and string formats
    if isinstance(mesh_shape, str):
        mesh_shape = ast.literal_eval(mesh_shape)

    if isinstance(mesh_shape, list) and len(mesh_shape) == 2:
        return tuple(mesh_shape)

    return None


def create_mesh_device(mesh_shape: Tuple[int, int], device_ids: Optional[list] = None) -> ttnn.MeshDevice:
    """
    Create a mesh device with the specified shape.

    Args:
        mesh_shape: Tuple of (rows, cols) for mesh shape
        device_ids: Optional list of device IDs (deprecated, not used by API)

    Returns:
        ttnn.MeshDevice instance
    """
    # Create mesh device with just the mesh shape
    # The API automatically selects available devices based on the mesh shape
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )


def compute_global_shape_from_shard(
    shard_shape: tuple,
    tensor_placement: Optional[Dict] = None,
) -> tuple:
    """
    Compute global tensor shape from per-shard shape using placement info.

    When a tensor is traced on a mesh with sharding, the stored shape is the
    per-shard shape. To recreate the tensor correctly, we need to compute the
    global shape by multiplying the sharded dimension by the distribution factor.

    Args:
        shard_shape: Per-shard tensor shape from traced config
        tensor_placement: Placement info with distribution_shape and placement

    Returns:
        Global tensor shape

    Example:
        shard_shape = (8, 197, 768)
        placement = "PlacementShard(0)", distribution_shape = "[2]"
        -> global_shape = (16, 197, 768)  # dim 0 multiplied by 2
    """
    if not tensor_placement:
        return shard_shape

    placement_str = tensor_placement.get("placement", "")

    # Only adjust for sharded placements
    if "PlacementShard" not in placement_str:
        return shard_shape

    # Parse shard dimension
    import re

    shard_dims = re.findall(r"PlacementShard\((\d+)\)", placement_str)
    if not shard_dims:
        return shard_shape

    shard_dim = int(shard_dims[-1])

    # Parse distribution shape to get the distribution factor
    distribution_shape_str = tensor_placement.get("distribution_shape", "")
    try:
        import ast

        distribution_shape = ast.literal_eval(distribution_shape_str)
        if isinstance(distribution_shape, list) and len(distribution_shape) > 0:
            # For PlacementShard(N), distribution_shape tells us how many parts
            # the tensor was split into along that dimension
            distribution_factor = distribution_shape[shard_dim] if shard_dim < len(distribution_shape) else 1
        else:
            distribution_factor = 1
    except (ValueError, SyntaxError, IndexError):
        distribution_factor = 1

    # Compute global shape by multiplying the sharded dimension
    global_shape = list(shard_shape)
    if shard_dim < len(global_shape) and distribution_factor > 1:
        global_shape[shard_dim] *= distribution_factor

    return tuple(global_shape)


def create_tensor_on_mesh(
    torch_tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig,
    tensor_placement: Optional[Dict] = None,
) -> ttnn.Tensor:
    """
    Create a TTNN tensor on a mesh device with optional placement.

    Args:
        torch_tensor: Input torch tensor (should be global shape, not per-shard)
        mesh_device: Mesh device to create tensor on
        dtype: TTNN data type
        layout: TTNN layout (TILE/ROW_MAJOR)
        memory_config: Memory configuration
        tensor_placement: Optional placement info from traced config

    Returns:
        TTNN tensor on mesh device with proper placement
    """
    # Determine mesh mapper based on placement
    if tensor_placement:
        placement_str = tensor_placement.get("placement", "")

        if "PlacementReplicate" in placement_str:
            # Replicate tensor across all devices
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        elif "PlacementShard" in placement_str:
            # Shard tensor across mesh
            # Parse shard dimension
            import re

            shard_dims = re.findall(r"PlacementShard\((\d+)\)", placement_str)
            shard_dim = int(shard_dims[-1]) if shard_dims else -1

            # Convert shard_dim to 2D mesh dims
            # For a 1D shard, we need to determine which mesh dimension to shard along
            # Based on mesh shape and shard dimension
            mesh_shape = tuple(mesh_device.shape)
            if mesh_shape[0] > 1 and mesh_shape[1] == 1:
                # Mesh is vertical (e.g., 2x1, 4x1) - shard along first dimension
                dims = (shard_dim, None)
            elif mesh_shape[0] == 1 and mesh_shape[1] > 1:
                # Mesh is horizontal (e.g., 1x2, 1x8) - shard along second dimension
                dims = (None, shard_dim)
            elif mesh_shape[0] > 1 and mesh_shape[1] > 1:
                # 2D mesh - use shard_dim for first mesh dimension
                dims = (shard_dim, None)
            else:
                # Single device (1x1) - no sharding
                dims = (None, None)

            mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)
        else:
            # Default to replicate if placement not recognized
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        # No placement info - default to replicate
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Create tensor on mesh
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


def get_mesh_shape() -> Optional[Tuple[int, int]]:
    """
    Get mesh shape from environment variable.

    Returns:
        Tuple of (rows, cols) or None if using single device

    Environment variable format:
        MESH_DEVICE_SHAPE="1x2" -> (1, 2)
        MESH_DEVICE_SHAPE="2x4" -> (2, 4)
        Not set -> None (use single device)
    """
    mesh_env = os.environ.get("MESH_DEVICE_SHAPE", "").strip()

    if not mesh_env:
        return None

    # Parse "NxM" format
    if "x" in mesh_env.lower():
        try:
            parts = mesh_env.lower().split("x")
            rows, cols = int(parts[0]), int(parts[1])
            return (rows, cols)
        except (ValueError, IndexError):
            print(f"⚠️ Invalid MESH_DEVICE_SHAPE format: {mesh_env}, expected NxM (e.g., 1x2)")
            return None

    return None


def mesh_tensor_to_torch(ttnn_tensor, mesh_device=None) -> torch.Tensor:
    """
    Convert a TTNN tensor (mesh or single device) to torch tensor.

    For mesh tensors, this extracts one device copy (typically device 0) since
    the model tracer tests expect the single-device result shape.

    Args:
        ttnn_tensor: TTNN tensor (mesh or single device)
        mesh_device: Optional mesh device reference (can be None)

    Returns:
        torch.Tensor: Converted tensor
    """
    # Check if this is a mesh tensor by checking the device attribute
    try:
        device = ttnn_tensor.device()
        # If device is a mesh device, extract tensor from first device
        if device is not None and hasattr(device, "get_num_devices"):
            # For mesh tensors, get the tensor from device 0 only
            # This avoids shape mismatches from concatenation
            device_tensors = ttnn.get_device_tensors(ttnn_tensor)
            if device_tensors and len(device_tensors) > 0:
                return ttnn.to_torch(device_tensors[0])
            # Fallback if get_device_tensors doesn't work
            return ttnn.to_torch(ttnn_tensor)
        else:
            # Single device tensor - direct conversion
            return ttnn.to_torch(ttnn_tensor)
    except Exception:
        # Fallback: try direct conversion (for host tensors or edge cases)
        return ttnn.to_torch(ttnn_tensor)
