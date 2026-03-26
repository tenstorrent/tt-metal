# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn


def _collect_submesh_fabric_ids(submesh, submesh_idx):
    """Return a list of (submesh_idx, local_coord, fabric_node_id) for every device in the submesh."""
    entries = []
    rows, cols = submesh.shape[0], submesh.shape[1]
    for row in range(rows):
        for col in range(cols):
            coord = ttnn.MeshCoordinate(row, col)
            fid = submesh.get_fabric_node_id(coord)
            entries.append((submesh_idx, (row, col), fid))
    return entries


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        }
    ],
    indirect=True,
)
def test_4x2_submeshes_on_single_galaxy(mesh_device):
    """Open a 8x4 mesh device and split it into 4 disjoint (4, 2) submeshes."""

    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(4, 2))

    assert len(submeshes) == 4, f"Expected 4 submeshes, got {len(submeshes)}"

    # Print fabric node IDs per submesh and build a chip_id -> submesh ownership map
    chip_to_submesh = {}
    for i, submesh in enumerate(submeshes):
        assert submesh.shape == ttnn.MeshShape(4, 2), f"Submesh {i} has shape {submesh.shape}, expected (4, 2)"
        assert submesh.get_num_devices() == 8, f"Submesh {i} has {submesh.get_num_devices()} devices, expected 8"

        entries = _collect_submesh_fabric_ids(submesh, i)
        for submesh_idx, local_coord, fid in entries:
            logger.info(f"Submesh {submesh_idx} coord {local_coord}: " f"mesh_id={fid.mesh_id}, chip_id={fid.chip_id}")
            chip_to_submesh[(int(fid.mesh_id), int(fid.chip_id))] = submesh_idx


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        }
    ],
    indirect=True,
)
def test_4x2_submeshes_on_pod(bh_2d_mesh_device):
    """Open a 4x8 mesh device and split it into 4 disjoint (4, 2) submeshes."""
    total_devices = bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]
    if total_devices < 32:
        pytest.skip(f"Test requires 32 devices (4x8 mesh), got {total_devices}")

    assert bh_2d_mesh_device.shape == ttnn.MeshShape(4, 8), f"Expected 4x8 mesh, got {bh_2d_mesh_device.shape}"

    submeshes = bh_2d_mesh_device.create_submeshes(ttnn.MeshShape(4, 2))

    assert len(submeshes) == 4, f"Expected 4 submeshes, got {len(submeshes)}"

    # Print fabric node IDs per submesh and build a chip_id -> submesh ownership map
    chip_to_submesh = {}
    for i, submesh in enumerate(submeshes):
        assert submesh.shape == ttnn.MeshShape(4, 2), f"Submesh {i} has shape {submesh.shape}, expected (4, 2)"
        assert submesh.get_num_devices() == 8, f"Submesh {i} has {submesh.get_num_devices()} devices, expected 8"

        entries = _collect_submesh_fabric_ids(submesh, i)
        for submesh_idx, local_coord, fid in entries:
            logger.info(f"Submesh {submesh_idx} coord {local_coord}: " f"mesh_id={fid.mesh_id}, chip_id={fid.chip_id}")
            chip_to_submesh[(int(fid.mesh_id), int(fid.chip_id))] = submesh_idx

    # Validate X-dim torus wraparound: [0,0]->[3,0] and [0,1]->[3,1] in each submesh.
    for i, submesh in enumerate(submeshes):
        for col in range(submesh.shape[1]):
            src_fid = submesh.get_fabric_node_id(ttnn.MeshCoordinate(0, col))
            dst_fid = submesh.get_fabric_node_id(ttnn.MeshCoordinate(3, col))
            connected = ttnn.are_fabric_neighbours(src_fid, dst_fid)
            logger.info(
                f"Submesh {i} [0,{col}]->[3,{col}]: "
                f"(mesh_id={src_fid.mesh_id}, chip_id={src_fid.chip_id}) -> "
                f"(mesh_id={dst_fid.mesh_id}, chip_id={dst_fid.chip_id}) "
                f"connected={connected}"
            )
            assert connected, (
                f"Submesh {i}: expected X-dim wraparound from [0,{col}] to [3,{col}] "
                f"(src={src_fid.mesh_id}:{src_fid.chip_id}, dst={dst_fid.mesh_id}:{dst_fid.chip_id})"
            )
