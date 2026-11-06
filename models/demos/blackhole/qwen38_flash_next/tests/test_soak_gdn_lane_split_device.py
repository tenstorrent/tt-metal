# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""k2_soak: QWEN38_GDN_LANE_SPLIT=1 (+ LS_HOIST, LS_CONVHEAD) gdn_step, 20 steps per trace, 1000 replays; the projection,
the older ring slots, the state, out and newest are rewritten every 10 replays.  Temporary: delete with soak_common.py."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401 (the demo's fixture)
from models.demos.blackhole.qwen38_flash_next.tests import soak_common as sc
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_gdn_step_cs_device import BF16, FP32, Case, up
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_gdn_step_ls_device import SRun, setenv
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
PARAMS = {**demo.DEVICE_PARAMS, "trace_region_size": 128 * 1024 * 1024}


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows,hoist,convhead", [(32, 1, 1), (16, 1, 1), (8, 1, 0), (32, 0, 0)], ids=lambda v: str(v))
def test_soak_gdn_lane_split(mesh_device, rows, hoist, convhead):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import lane_split as ls

    mesh = mesh_device
    assert gs.HEADS == 6, "run with MESH_DEVICE=P150x8"
    grid = mesh.compute_with_storage_grid_size()
    assert ls.plan_for(rows, grid.x, grid.y, vb=1, g="auto", hoist=bool(hoist), convhead=bool(convhead)) is not None
    case = Case(rows, 500 + rows)
    setenv("1", 1, "auto", hoist, convhead)
    run = SRun(mesh, case)
    projected = up(mesh, case.projections[0], BF16)
    older = [up(mesh, t, BF16) for t in case.older[0]]
    out = up(mesh, run.sent_out, BF16)
    newest = up(mesh, run.sent_newest, BF16)
    state_host = torch.cat([case.state, run.extra_lane])
    hosts = []
    for v in range(sc.VARIANTS):
        i = v * 7  # projections / slots of step 0, 7, 14
        hosts.append(
            dict(
                proj=sc.host_tensor(case.projections[i], BF16, ttnn.TILE_LAYOUT, mesh),
                older=[sc.host_tensor(t, BF16, ttnn.TILE_LAYOUT, mesh) for t in case.older[i]],
            )
        )
    h_state = sc.host_tensor(state_host, FP32, ttnn.TILE_LAYOUT, mesh)
    h_out = sc.host_tensor(run.sent_out, BF16, ttnn.TILE_LAYOUT, mesh)
    h_newest = sc.host_tensor(run.sent_newest, BF16, ttnn.TILE_LAYOUT, mesh)

    def call(keep):
        gs.run(projected, older, newest, run.taps, run.constants, run.norm, run.state, out)

    def prepare(i):
        h = hosts[i % sc.VARIANTS]
        sc.copy_in(h["proj"], projected)
        for hh, d in zip(h["older"], older):
            sc.copy_in(hh, d)
        sc.copy_in(h_state, run.state)
        sc.copy_in(h_out, out)
        sc.copy_in(h_newest, newest)

    def observe():
        return [d for t in (out, newest, run.state) for d in sc.dies(t)]

    try:
        sc.soak(mesh, f"gdn_lane_split_h{hoist}_ch{convhead}", rows, call, prepare, observe)
    finally:
        setenv("0")
