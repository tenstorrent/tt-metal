# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""k2_soak: QWEN38_ROUTER_SC_LANES=1 moe_dense, 20 calls per trace, 1000 replays, logits re-randomized every 10 replays.
Temporary: delete with soak_common.py."""

import os

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import soak_common as sc
from models.demos.blackhole.qwen38_flash_next.tests.test_router_sc_lanes_device import FLAG, MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.tests.test_router_sc_lanes_device import PARAMS as BASE_PARAMS
from models.demos.blackhole.qwen38_flash_next.tests.test_router_sc_lanes_device import (
    _logit_rows,
    _moe_dense,
    _outputs,
    _upload,
    rt,
    rt_sc,
)

PARAMS = {**BASE_PARAMS, "trace_region_size": 128 * 1024 * 1024}


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [32, 17, 9])
def test_soak_router_lanes(mesh_device, rows):
    mesh = mesh_device
    rt_sc.prepare(mesh)
    variants = [_logit_rows(rows, 9000 + rows + 100 * v)[0].reshape(1, 1, rows, rt.EXPERTS) for v in range(sc.VARIANTS)]
    hosts = [sc.host_tensor(v, ttnn.float32, ttnn.TILE_LAYOUT, mesh) for v in variants]
    logits = _upload(mesh, variants[0])
    outs = _outputs(mesh, rows)

    def call(keep):
        for t in _moe_dense(mesh, rows, logits, True, outs)[2:]:
            if t is not None:
                ttnn.deallocate(t)

    def prepare(i):
        sc.copy_in(hosts[i % sc.VARIANTS], logits)

    def observe():
        return [d for t in outs for d in sc.dies(t)]

    try:
        sc.soak(mesh, "router_lanes", rows, call, prepare, observe)
    finally:
        os.environ.pop(FLAG, None)
