# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re
import torch
import ttnn
from pathlib import Path
from typing import Optional


def get_mesh_shape_from_textproto(textproto_path: str, mesh_id: int = 0) -> Optional[ttnn.MeshShape]:
    """Parse mesh shape from textproto file for a specific mesh_id."""
    path = Path(textproto_path)

    # Try multiple path resolution strategies
    if not path.is_absolute():
        # First try as-is (relative to current working directory)
        if not path.exists():
            # Try relative to TT_METAL_HOME
            tt_metal_home = os.environ.get("TT_METAL_HOME")
            if tt_metal_home:
                candidate = Path(tt_metal_home) / path
                if candidate.exists():
                    path = candidate
                else:
                    # Try relative to repo root (common when running from repo root)
                    cwd = Path.cwd()
                    candidate = cwd / path
                    if candidate.exists():
                        path = candidate

    if not path.exists():
        return None

    with open(path, "r") as f:
        content = f.read()

    # Find mesh descriptor referenced by mesh_id in instances
    # Look for: instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    mesh_desc_pattern = rf'instances\s*{{\s*mesh\s*{{\s*mesh_descriptor:\s*"([^"]+)"\s+mesh_id:\s+{mesh_id}\s*}}\s*}}'
    match = re.search(mesh_desc_pattern, content)
    if not match:
        return None

    mesh_descriptor_name = match.group(1)

    # Find device_topology for this mesh descriptor
    # Look for: mesh_descriptors { name: "M0" ... device_topology { dims: [ x, y ] } }
    mesh_pattern = rf'mesh_descriptors\s*{{\s*name:\s*"{re.escape(mesh_descriptor_name)}"[^}}]*device_topology\s*{{\s*dims:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]'
    match = re.search(mesh_pattern, content, re.DOTALL)
    if not match:
        return None

    return ttnn.MeshShape(int(match.group(1)), int(match.group(2)))


def compute_split_mesh_shape(full_shape: ttnn.MeshShape) -> Optional[ttnn.MeshShape]:
    if full_shape[0] >= full_shape[1]:
        if full_shape[0] % 2 == 0:
            return ttnn.MeshShape(full_shape[0] // 2, full_shape[1])
        return None
    else:
        if full_shape[1] % 2 == 0:
            return ttnn.MeshShape(full_shape[0], full_shape[1] // 2)
        return None


def run_multiprocess_pipeline(mesh_shape=None):
    torch.manual_seed(0)

    cluster_type = ttnn.cluster.get_cluster_type()
    supported_cluster_types = [
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        ttnn.cluster.ClusterType.P300,
        ttnn.cluster.ClusterType.P300_X2,
        ttnn.cluster.ClusterType.CUSTOM,
    ]

    if cluster_type not in supported_cluster_types:
        num_devices = ttnn._ttnn.device.GetNumAvailableDevices()
        num_pci_devices = ttnn._ttnn.device.GetNumPCIeDevices()
        raise ValueError(
            f"Multi-mesh workloads are not yet supported on {cluster_type} systems.\n"
            f"Multi-mesh requires all chips to have direct PCI access for process isolation via TT_VISIBLE_DEVICES.\n"
            f"This allows splitting the full mesh into smaller sub-meshes (e.g., Galaxy 8x4 → 2x 4x4 meshes).\n"
            f"Current system: {num_devices} total devices, {num_pci_devices} PCI-accessible devices.\n"
            f"Supported systems: Galaxy (all 32 chips have PCI, can split into multiple meshes), "
            f"P300 (all chips have PCI, can split into multiple meshes).\n"
            f"Unsupported examples: N300 (only r-chip has PCI, l-chip is ethernet-only), "
            f"T3000 (only 4 of 8 chips have PCI, others are ethernet-only)."
        )

    if mesh_shape is None:
        mesh_graph_path = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        if mesh_graph_path:
            # tt-run always sets TT_MESH_GRAPH_DESC_PATH from the rank binding file's mesh_graph_desc_path.
            # The mesh shape is already defined per mesh in the textproto.
            mesh_id = int(os.environ.get("TT_MESH_ID", "0"))
            mesh_shape = get_mesh_shape_from_textproto(mesh_graph_path, mesh_id)

        if mesh_shape is None:
            # Fallback: get full mesh shape and split it (used when mesh_graph_path doesn't exist or file is missing)
            full_mesh_shape = ttnn.cluster.get_mesh_shape()
            num_devices = full_mesh_shape[0] * full_mesh_shape[1]

            if num_devices < 2:
                raise ValueError(f"Not enough devices: have {num_devices}, need at least 2")

            mesh_shape = compute_split_mesh_shape(full_mesh_shape)
            if mesh_shape is None:
                raise ValueError(f"Cannot evenly split {full_mesh_shape[0]}x{full_mesh_shape[1]} mesh into 2 parts")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized")
    if int(ttnn.distributed_context_get_size()) != 2:
        raise ValueError("This test requires 2 processes to run")

    core_coord = ttnn.CoreCoord(0, 0)
    socket_connections = [
        ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, core_coord), ttnn.MeshCoreCoord(coord, core_coord))
        for coord in ttnn.MeshCoordinateRange(mesh_shape)
    ]

    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config, 0, 1)

    torch_input = torch.randn(1, 1, 1024, 1024, dtype=torch.float32)
    num_devices_in_mesh = mesh_shape[0] * mesh_shape[1]

    if num_devices_in_mesh > 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(2, 3), mesh_shape=mesh_shape)
        mesh_composer = ttnn.ConcatMesh2dToTensor(device, mesh_shape=mesh_shape, dims=(2, 3))
    else:
        mesh_mapper = None
        mesh_composer = None

    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)

    rank = int(ttnn.distributed_context_get_rank())

    if rank == 0:
        send_socket = ttnn.MeshSocket(device, socket_config)
        ttnn.experimental.send_async(ttnn.relu(ttnn_input), send_socket)
        print(f"[Rank {rank}] Sent ReLU output")
    else:
        recv_socket = ttnn.MeshSocket(device, socket_config)
        upstream_input = ttnn.allocate_tensor_on_device(ttnn_input.spec, device)
        ttnn.experimental.recv_async(upstream_input, recv_socket)
        print(f"[Rank {rank}] Received data")

        expected = torch.exp(torch.nn.functional.relu(torch_input))
        torch_tensor = ttnn.to_torch(
            ttnn.from_device(ttnn.exp(upstream_input)),
            mesh_composer=mesh_composer,
        )

        if torch.allclose(torch_tensor, expected, rtol=1e-3, atol=1e-3):
            print(f"[Rank {rank}] Test passed: Output matches expected")
        else:
            raise ValueError(f"[Rank {rank}] Test failed: Output mismatch")

    ttnn.distributed_context_barrier()
    ttnn.close_device(device)
