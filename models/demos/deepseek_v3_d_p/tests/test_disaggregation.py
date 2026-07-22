# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for KvChunkAddressTable Python bindings.
This test verifies that the disaggregation APIs are accessible from Python
and work correctly.
"""

import socket

import pytest
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4), (8, 4), (1, 2)],
    ids=["32x4", "8x4", "1x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_fnids(mesh_device):
    mesh_shape = list(mesh_device.shape)

    host_name = socket.gethostname()
    logger.info(f"Host name: {host_name}")

    rank = ttnn.distributed_context_get_rank()
    size = ttnn.distributed_context_get_size()

    total_rows = mesh_shape[0]
    rank_row_start = int(rank) * total_rows // int(size)
    rank_row_end = rank_row_start + total_rows // int(size)

    logger.info(f"Rank: {rank}, Size: {size}, Row start: {rank_row_start}, Row end: {rank_row_end}")

    all_fabric_node_ids = []
    for row in range(rank_row_start, rank_row_end):
        fabric_node_ids = []
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            fabric_node_ids.append(fabric_node_id)

        all_fabric_node_ids.extend(fabric_node_ids)
        for idx, fid in enumerate(fabric_node_ids):
            mesh_id = int(fid.mesh_id)
            chip_id = int(fid.chip_id)
            logger.info(
                f"  Node {idx} row={row}, col={col}: mesh_id={mesh_id},  chip_id={chip_id} host_name={host_name}"
            )
