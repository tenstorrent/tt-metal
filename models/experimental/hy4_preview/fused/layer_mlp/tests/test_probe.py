# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the 8x4 mesh tells the MLP layer program's builder: the worker grid, the DRAM bank anchors, the MoE piece's
placement on a mesh device and the L1 its persistent tensors take. Mock or board; log lines start with HY4_MK8."""

import pytest
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe.program import RoutedExperts, bank_anchors
from models.experimental.hy4_preview.tt import prefill_bridge as pb

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), (8, 4))], indirect=True, ids=["8x4"]
)


def l1(mesh_device):
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    return v.total_bytes_per_bank, v.total_bytes_allocated_per_bank


@galaxy
def test_probe_grid(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"HY4_MK8 probe grid {grid.x}x{grid.y}, L1 per core (total, allocated) {l1(mesh_device)}")
    anchors = bank_anchors(mesh_device)
    logger.info(f"HY4_MK8 probe bank anchors {[(a.x, a.y) for a in anchors]}")
    noc = {
        (x, y): mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        for x in range(grid.x)
        for y in range(grid.y)
    }
    logger.info(
        f"HY4_MK8 probe noc x of logical columns {[noc[(x, 0)].x for x in range(grid.x)]}, y of rows {[noc[(0, y)].y for y in range(grid.y)]}"
    )
    for G, D in ((6, 6), (5, 4), (4, 4), (4, 3)):
        before = l1(mesh_device)[1]
        op = RoutedExperts(
            mesh_device,
            L.tp_share(32),
            gate_up_per_bank=G,
            down_per_bank=D,
            router=True,
            n_plain=1,
            plain_id0=256,
            plain_terms=2,
        )
        used = 1 + op.NF + len(op.r_cores)
        logger.info(
            f"HY4_MK8 probe MoE G={G} D={D}: hub {(op.hub.x, op.hub.y)} fin {[(c.x, c.y) for c in op.fin]} router "
            f"{[(c.x, c.y) for c in op.r_cores]} free {[(c.x, c.y) for c in op.placement.spare[used:]]}; "
            f"L1 tensors {l1(mesh_device)[1] - before} B; sync at {op.sync.buffer_address():#x}"
        )
        logger.info(f"HY4_MK8 probe MoE G={G} D={D} readers {op.placement.describe()}")


@galaxy
def test_probe_anchors(mesh_device):
    """The DRAM banks' nearest worker cores chip by chip: the MoE piece places its readers by chip (0, 0)'s."""
    rows, cols = tuple(mesh_device.shape)
    seen = {}
    for r in range(rows):
        for c in range(cols):
            try:
                a = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(
                    ttnn.NOC.NOC_0, ttnn.MeshCoordinate(r, c)
                )
            except TypeError as e:
                logger.info(f"HY4_MK8 probe anchors: no per-chip query ({e})")
                return
            key = tuple((v.x, v.y) for v in ([a[b] for b in sorted(a)] if isinstance(a, dict) else list(a)))
            seen.setdefault(key, []).append((r, c))
    for key, chips in seen.items():
        logger.info(f"HY4_MK8 probe anchors {list(key)}: {len(chips)} chips {chips if len(chips) < 32 else 'all'}")
