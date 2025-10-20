#!/usr/bin/env python3
"""
Demo: Automatically compose multi-device sharded tensors using TensorTopology

This script shows how to infer the correct MeshToTensor composer from a sharded
ttnn.Tensor's topology metadata and use it to compose shards on host.

It validates both host-sharded and device-sharded cases.

Usage:
  python scripts/ttnn_auto_compose_demo.py

Notes:
  - Requires a multi-device setup to exercise true sharding (>=2 devices).
  - If only one device is available, the script still runs but will skip
    device-sharded assertions.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import ttnn


def _infer_dims_from_placements(placements: List[object]) -> List[int]:
    """
    Build the MeshComposerConfig.dims vector based on placements.

    Rule:
    - For PlacementShard(dim), use that tensor dim.
    - For PlacementReplicate(), default to dim 0 (stack along the outer axis).

    This mirrors test patterns like dims={0,2,1} in 3D where first dist dim
    is replicated and mapped to the outer concat dimension.
    """
    dims: List[int] = []
    for p in placements:
        if isinstance(p, ttnn.PlacementShard):
            dims.append(p.dim)
        else:
            # Replication: choose outer-dim stacking by default
            dims.append(0)
    return dims


def infer_mesh_composer_from_topology(
    tensor: ttnn.Tensor, mesh_device: Optional[ttnn.MeshDevice] = None
) -> Optional[ttnn.CppMeshToTensor]:
    """
    Return a MeshToTensor composer inferred from the tensor's TensorTopology,
    or None if no composition is needed (fully replicated, single-device).
    """
    topology = tensor.tensor_topology()
    placements = topology.placements()
    dist_shape = list(topology.distribution_shape())

    # If there is no distribution dimension info, or size==1, nothing to compose
    if len(dist_shape) == 0 or (len(dist_shape) == 1 and dist_shape[0] == 1):
        return None

    # Prefer mesh_device from the tensor itself if present
    device = tensor.device() if hasattr(tensor, "device") else None
    # tensor.device() returns None if the tensor is on the host (ttnn/core/tensor/tensor.cpp --> Tensor::device())
    if device is None:
        device = mesh_device

    if device is None:
        raise RuntimeError("infer_mesh_composer_from_topology: mesh_device is required when tensor has no device")

    # 1D: use the convenience composer
    assert len(dist_shape) == len(
        placements
    )  # guaranteed by TT_FATAL check in ttnn/core/distributed/distributed_tensor.cpp
    if len(dist_shape) == 1:
        shard_dim = None
        p = placements[0]
        if isinstance(p, ttnn.PlacementShard):
            shard_dim = p.dim

        if shard_dim is None:
            # Fully replicated across 1D mesh; no concatenation necessary
            return None

        return ttnn.concat_mesh_to_tensor_composer(device, shard_dim)

    # ND: build dims vector and mesh_shape_override from topology
    dims = _infer_dims_from_placements(placements)
    composer_cfg = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=topology.distribution_shape())
    return ttnn.create_mesh_composer(device, composer_cfg)


def to_torch_auto_compose(tensor: ttnn.Tensor, mesh_device: Optional[ttnn.MeshDevice] = None) -> torch.Tensor:
    """
    Convert a (possibly multi-device) TTNN tensor to torch, automatically
    composing shards based on the tensor's topology.
    """

    composer = infer_mesh_composer_from_topology(tensor, mesh_device=mesh_device)
    return tensor.to_torch(mesh_composer=composer)


def _make_known_pattern(num_chunks: int) -> torch.Tensor:
    # Produces shape [num_chunks, 1, 3, 1] with per-chunk distinct values
    #   chunk i -> [i*1, i*2, i*3]
    rows = []
    for i in range(num_chunks):
        rows.append(torch.tensor([[[i * 1.0], [i * 2.0], [i * 3.0]]]).transpose(0, 1))  # [1,3,1]
    data = torch.stack(rows, dim=0)  # [num_chunks,1,3,1]
    return data.to(torch.bfloat16)


def test_host_sharded_1d(mesh_device: ttnn.MeshDevice, num_devices: int) -> None:
    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Create a host-sharded tensor along dim=0
    tt_host_sharded = ttnn.from_torch(
        torch_in, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    # Validate that auto-composition returns original torch input
    torch_auto = to_torch_auto_compose(tt_host_sharded, mesh_device)

    # Reference using explicit composer
    torch_ref = ttnn.to_torch(tt_host_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch"
    assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch on host-sharded tensor"


def test_device_sharded_1d(mesh_device: ttnn.MeshDevice, num_devices: int) -> None:
    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Distribute to mesh device directly (device storage)
    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0)
    tt_host = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16)
    tt_dev_sharded = ttnn.distribute_tensor(tt_host, mapper, mesh_device)

    # Validate that auto-composition returns original torch input
    torch_auto = to_torch_auto_compose(tt_dev_sharded, mesh_device)

    # Reference using explicit composer through high-level API
    torch_ref = ttnn.to_torch(tt_dev_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch (device-sharded)"
    assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch on device-sharded tensor"


def main() -> int:
    # default to N150 and configurable via MESH_DEVICE environment variable
    mesh_shape = {
        "N150": [1, 1],
        "N300": [1, 2],
        "N150x4": [1, 4],
        "T3K": [1, 8],
        "TG": [8, 4],
        "P150": [1, 1],
        "P300": [1, 2],
        "P150x4": [1, 4],
        "P150x8": [1, 8],
    }.get(os.environ.get("MESH_DEVICE"), [1, 1])

    # Use a context manager so the device is closed on exit
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape))
    print(f"Opened MeshDevice with shape: {mesh_device.shape}")

    if (num_devices := mesh_device.get_num_devices()) < 2:  # noqa: E741
        print("Only one device available — skipping tests.")
    else:
        print(f"Running 1D host-sharded test across {num_devices} devices...")
        test_host_sharded_1d(mesh_device, num_devices)
        print("Host-sharded composition OK.")

        print(f"Running 1D device-sharded test across {num_devices} devices...")
        test_device_sharded_1d(mesh_device, num_devices)
        print("Device-sharded composition OK.")

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
