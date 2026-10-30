# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: where the x all-gather's port cores sit vs its speed. K3's head gather (640 rows x H / TP 1792 bf16 a chip,
ring of 4, 2 links) with fabric_all_gather_py's ports in (A) the head's mover column (logical 11: every port enters
the Ethernet row at one x on NOC1 - the funnel), (B) anywhere ("auto": each port directly below its Ethernet core),
(C) the other edge column (logical 0). PPL_CASES picks; PPL_ROWS / PPL_COLS the shard."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.moe_block.tests.test_fused_head_parts import _fab
from models.demos.moe_block.tests.test_tp_collectives import per_chip_programs

R, C = (int(v) for v in os.environ.get("MOE_MESH", "1x4").split("x"))
ROWS, COLS = int(os.environ.get("PPL_ROWS", "640")), int(os.environ.get("PPL_COLS", "1792"))


@pytest.mark.parametrize("mesh_device, device_params", [((R, C), _fab(l1_small_size=0))], indirect=True)
def test_port_place(mesh_device, device_params):
    from ttnn.bringup.combine_rf.fabric_all_gather_py import fabric_all_gather, plan

    grid = mesh_device.compute_with_storage_grid_size()
    col = lambda x: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, grid.y - 1))])  # noqa
    cases = {
        "A_mover_col": dict(sub_core_grid=col(grid.x - 1), placement="simple"),
        "A2_mover_col_auto": dict(sub_core_grid=col(grid.x - 1), placement="auto"),
        "B_under_eth": dict(sub_core_grid=None, placement="auto"),
        "C_left_col": dict(sub_core_grid=col(0), placement="simple"),
    }
    sel = os.environ.get("PPL_CASES")
    # (the shard as [COLS, ROWS]: the op gathers along rows - the same bytes as the head's [ROWS, COLS] width gather)
    x = ttnn.from_torch(
        torch.randn(C * COLS, ROWS).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(None, 0)),
    )
    virt = mesh_device.worker_core_from_logical_core
    for name, kw in cases.items():
        if sel and name not in sel.split(","):
            continue
        chips, _ = plan(mesh_device, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=2, balance=True, **kw)
        ports = {k: (virt(c).x, virt(c).y) for k, c in next(iter(chips.values()))["ports"].items()}
        out = None
        ts = []
        for it in range(8):
            out = fabric_all_gather(
                x, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=2, dim=0, balance=True, output=out, **kw
            )
            ttnn.synchronize_device(mesh_device)
            progs = [r for r in per_chip_programs(mesh_device) if r]
            if it >= 2:
                ts.append(max(sum(r) for r in progs))
        b = 3 * ROWS * COLS * 2  # (bytes in per chip)
        us = statistics.median(ts)
        logger.info(
            f"PPL {name:18s}: {us:6.1f} us ({b / us / 1e3:5.1f} GB/s in a chip; ring floor {b / 2 / 2 / 25e3:.0f} us "
            f"at 25 GB/s a link); ports (chip 0, NoC xy) {ports}"
        )
