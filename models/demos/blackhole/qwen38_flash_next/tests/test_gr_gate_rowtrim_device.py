# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of QWEN38_GR_GATE_ROWTRIM: gate3 with the flag off and on, same inputs, rows 0..rows-1 of the block and the
injection bit for bit (program cache off so each call rebuilds the program), then the traced slope per call off vs on.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=... pytest this_file -s"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.tests.test_gr_gate_device import (
    DEVICE_PARAMS,
    bits,
    host_valid,
    make_inputs,
    traced_us,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("GG_STEPS", "10"))
RESULTS = os.environ.get("GG_RESULTS", "/tmp/gg_rowtrim_results.jsonl")


def run(flag, p, n, u, rows, mesh):
    os.environ[gate3.ROWTRIM_ENV] = "1" if flag else "0"
    o = gate3.low_rank_gate3(p, n, u, matmul="chain")
    ttnn.synchronize_device(mesh)
    got = host_valid(o, rows)
    for t in o:
        ttnn.deallocate(t)
    return got


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 4, 5, 6, 8])
def test_rowtrim_bitwise_and_slope(mesh_device, rows):
    mesh = mesh_device
    try:
        for step in range(STEPS):
            p, n, u = make_inputs(mesh, rows, 100 * rows + step)
            off, on = run(False, p, n, u, rows, mesh), run(True, p, n, u, rows, mesh)
            for d, (a, b) in enumerate(zip(off, on)):
                for what, x, y in zip(("block", "injection"), a, b):
                    xb, yb = bits(x), bits(y)
                    bad = (
                        sorted({int(r) for r, _ in (xb != yb).reshape(xb.shape[-2], -1).nonzero().tolist()})
                        if xb.shape == yb.shape
                        else "shape"
                    )
                    assert torch.equal(xb, yb), f"rows={rows} step {step} dev {d}: {what} differs, rows {bad}"
            for t in (p, n, u):
                ttnn.deallocate(t)
        logger.info(f"rows={rows}: flag off/on bit-identical over {STEPS} steps")
        p, n, u = make_inputs(mesh, rows, 7)
        times = {"off": [], "on": []}
        mesh.enable_program_cache()
        for rep in range(3):
            for name, flag in (("off", False), ("on", True)):
                os.environ[gate3.ROWTRIM_ENV] = "1" if flag else "0"
                mesh.clear_program_cache()
                times[name].append(traced_us(mesh, lambda: gate3.low_rank_gate3(p, n, u, matmul="chain")))
                logger.info(f"rows={rows} rep {rep} {name}: slope/call {times[name][-1][0]} us")
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(rows=rows, us=times)) + "\n")
    finally:
        os.environ.pop(gate3.ROWTRIM_ENV, None)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_zone_profile(mesh_device):
    """Study build (QWEN38_FUSED_ZONES=1 TT_METAL_DEVICE_PROFILER=1, QWEN38_GR_GATE_ROWTRIM = 0 or 1, rows 1): gate3 a few
    times, then the device profiler is read; the fz_gr_* zones land in <profiler dir>/.logs/profile_log_device.csv."""
    mesh = mesh_device
    mesh.enable_program_cache()
    p, n, u = make_inputs(mesh, 1, 5)
    for _ in range(3):
        for t in gate3.low_rank_gate3(p, n, u, matmul="chain"):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    for _ in range(5):
        for t in gate3.low_rank_gate3(p, n, u, matmul="chain"):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
