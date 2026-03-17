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
        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw

        # Check if it's a replicate placement
        if "PlacementReplicate" in placement_str:
            return ttnn.ReplicateTensorToMesh

        # Check if it's a shard placement
        if "PlacementShard" in placement_str:
            # Extract shard dimensions
            # e.g., "['PlacementShard(2)', 'PlacementShard(3)']" -> shard on dims 2,3
            import re

            shard_dims = re.findall(r"PlacementShard\((-?\d+)\)", placement_str)

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


def _parse_shard_dim(placement_str: str) -> int:
    """Extract shard dimension from placement string, handling negative dims."""
    import re

    shard_dims = re.findall(r"PlacementShard\((-?\d+)\)", placement_str)
    return int(shard_dims[-1]) if shard_dims else -1


def _is_shard_placement(tensor_placement: Optional[Dict], num_devices: int) -> bool:
    """Check if placement is a shard placement with multiple devices."""
    if not tensor_placement:
        return False
    placement_str = tensor_placement.get("placement", "")
    # If it has both Replicate and Shard, check which comes first or treat as replicate
    if "PlacementReplicate" in placement_str and "PlacementShard" not in placement_str:
        return False
    return "PlacementShard" in placement_str and num_devices > 1


def get_mesh_composer(mesh_device, tensor_placement: Optional[Dict] = None):
    """
    Create a mesh composer matching the tensor placement for converting back to torch.

    For sharded tensors, returns a ConcatMesh2dToTensor that reassembles shards.
    For replicated tensors, returns None (caller should use device 0 extraction).

    Args:
        mesh_device: The mesh device
        tensor_placement: Placement info from traced config

    Returns:
        Mesh composer or None
    """
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if not _is_shard_placement(tensor_placement, num_devices):
        return None

    placement_str = tensor_placement.get("placement", "")
    shard_dim = _parse_shard_dim(placement_str)

    try:
        mesh_shape = (1, num_devices)
        return ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, dims=shard_dim)
    except (TypeError, RuntimeError):
        return None


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
        torch_tensor: Input torch tensor
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
        import re
        import ast as _ast

        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw

        mesh_shape_raw = tensor_placement.get("mesh_device_shape", "[1, 1]")
        if isinstance(mesh_shape_raw, str):
            mesh_shape_raw = _ast.literal_eval(mesh_shape_raw)
        mesh_shape_tuple = tuple(mesh_shape_raw) if isinstance(mesh_shape_raw, (list, tuple)) else (1, 1)

        # Check if the actual device mesh can support the traced mesh shape.
        # If not (e.g., traced on Galaxy 4x8 but running on N150 1x1), fall back to replicate.
        try:
            actual_mesh = mesh_device.shape
            actual_rows, actual_cols = actual_mesh[0], actual_mesh[1]
        except Exception:
            actual_rows, actual_cols = 1, 1
        traced_rows = mesh_shape_tuple[0]
        traced_cols = mesh_shape_tuple[1] if len(mesh_shape_tuple) > 1 else 1
        mesh_compatible = actual_rows >= traced_rows and actual_cols >= traced_cols

        entries = re.findall(r"Placement(?:Shard\(-?\d+\)|Replicate)", placement_str)

        if not mesh_compatible or not entries or "PlacementShard" not in placement_str:
            # Device mesh too small or no shard placement - replicate
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        elif len(entries) >= 2:
            dims = []
            for entry in entries[:2]:
                shard_match = re.search(r"PlacementShard\((-?\d+)\)", entry)
                if shard_match:
                    dims.append(int(shard_match.group(1)))
                else:
                    dims.append(None)
            dims_tuple = tuple(dims)

            # Traced shapes are per-device (post-shard). ShardTensor2dMesh expects
            # global (pre-shard) shapes. Expand by tiling shard dims by mesh sizes.
            repeat_factors = [1] * torch_tensor.ndim
            if dims_tuple[0] is not None:
                repeat_factors[dims_tuple[0]] = mesh_shape_tuple[0]
            if dims_tuple[1] is not None:
                repeat_factors[dims_tuple[1]] = mesh_shape_tuple[1]
            torch_tensor = torch_tensor.repeat(*repeat_factors)

            mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims_tuple, mesh_shape=mesh_shape_tuple)
        elif len(entries) == 1:
            shard_match = re.search(r"PlacementShard\((-?\d+)\)", entries[0])
            if shard_match:
                dim = int(shard_match.group(1))
                dims_tuple = (None, dim)

                repeat_factors = [1] * torch_tensor.ndim
                repeat_factors[dim] = mesh_shape_tuple[1] if len(mesh_shape_tuple) > 1 else mesh_shape_tuple[0]
                torch_tensor = torch_tensor.repeat(*repeat_factors)

                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims_tuple, mesh_shape=mesh_shape_tuple)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
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


def mesh_tensor_to_torch(ttnn_tensor, mesh_device=None, mesh_composer=None) -> torch.Tensor:
    """
    Convert a TTNN tensor (mesh or single device) to torch tensor.

    For replicated mesh tensors, extracts device 0's copy.
    For sharded mesh tensors, uses the provided mesh_composer to reassemble
    all device shards into the original full tensor.

    Args:
        ttnn_tensor: TTNN tensor (mesh or single device)
        mesh_device: Optional mesh device reference (can be None)
        mesh_composer: Optional mesh composer for reassembling sharded tensors.
                       If None, falls back to extracting device 0 only.

    Returns:
        torch.Tensor: Converted tensor
    """
    # Check if this is a mesh tensor by checking the device attribute
    try:
        device = ttnn_tensor.device()
        # If device is a mesh device, handle appropriately
        if device is not None and hasattr(device, "get_num_devices"):
            # If a mesh_composer is provided, use it to reassemble shards
            if mesh_composer is not None:
                return ttnn.to_torch(ttnn_tensor, mesh_composer=mesh_composer)
            # Default: extract device 0 only (works for replicated tensors)
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
