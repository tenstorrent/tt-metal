# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the mesh the model opens tells Python about its fabric: payload, links, forwarding directions, the worker
grid, where global semaphores land. Runs on the mock Galaxy and on the board; asserts nothing beyond what the
collective relies on, and logs the rest (lines start with HY4_MK1)."""

import pytest
from loguru import logger

import ttnn
from models.experimental.hy4_preview.tt import prefill_bridge as pb

GALAXY_MESH = (8, 4)

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), GALAXY_MESH)], indirect=True, ids=["8x4-torus"]
)


@galaxy
def test_probe_fabric(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"HY4_COLLECTIVE probe mesh {rows}x{cols} grid {grid.x}x{grid.y} "
        f"max_payload {ttnn.get_tt_fabric_max_payload_size_bytes()} header {ttnn.get_tt_fabric_packet_header_size_bytes()}"
    )
    node = lambda r, c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    src = node(0, 0)
    logger.info(
        f"HY4_COLLECTIVE probe node(0,0) = mesh {src.mesh_id} chip {src.chip_id}; node(1,0) chip {node(1, 0).chip_id}"
    )
    for r, c in [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (7, 0), (1, 1), (4, 2)]:
        dst = node(r, c)
        direction = ttnn.get_eth_forwarding_direction(src, dst)
        links = ttnn.get_forwarding_link_indices(mesh_device, src, dst)
        logger.info(
            f"HY4_COLLECTIVE probe (0,0)->({r},{c}) chip {dst.chip_id}: direction {direction} links {list(links)}"
        )
    for xy in [(0, 0), (1, 0), (0, 1), (5, 3)]:
        v = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        logger.info(f"HY4_COLLECTIVE probe logical core {xy} -> noc ({v.x},{v.y})")
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    for kind in (ttnn.BufferType.L1_SMALL, ttnn.BufferType.L1):
        sem = ttnn.create_global_semaphore(mesh_device, cores, 0, kind)
        logger.info(
            f"HY4_COLLECTIVE probe global semaphore in {kind}: address {ttnn.get_global_semaphore_address(sem):#x}"
        )
    for kind in (ttnn.BufferType.L1_SMALL, ttnn.BufferType.L1):
        view = ttnn.get_memory_view(mesh_device, kind)
        logger.info(
            f"HY4_COLLECTIVE probe {kind}: per bank {view.total_bytes_per_bank} allocated {view.total_bytes_allocated_per_bank}"
        )
