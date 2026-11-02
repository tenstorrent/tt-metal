# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of QWEN38_FRONT_ROWTRIM: the GR front v3 with the flag off and on, same inputs; the normalized' tensor and the
gathered partials, rows 0..rows-1 of every device, bit for bit (the gathered stats pages are not written by v3, as in
test_gr_front_device), then the traced slope per call off vs on.  The program cache is on (as in test_gr_front_device: the v3 transport's call counter is only reproducible with it; the
defines are part of the program hash, so each flag value is its own cached program).
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=... pytest this_file -s"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.tests.test_gr_front_device import (
    DEVICE_PARAMS,
    make_inputs,
    traced_us,
    valid_rows,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front3

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("GF_STEPS", "10"))
RESULTS = os.environ.get("GF_RESULTS", "/tmp/gf_rowtrim_results.jsonl")


def run(flag, res, gam, down, rows, mesh):
    os.environ[front3.ROWTRIM_ENV] = "1" if flag else "0"
    o = front3.stats_recip_down_gather(res, gam, down)
    ttnn.synchronize_device(mesh)
    got = valid_rows(o, rows)
    for t in o:
        ttnn.deallocate(t)
    return got


def bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def same(a_runs, b_runs, report=None):
    """Normalized' and partials of every device equal bit for bit (the gathered stats pages are not written by v3)."""
    for d, (a, b) in enumerate(zip(a_runs, b_runs)):
        for what, x, y in zip(("stats", "normalized", "partials"), a, b):
            if what != "stats" and not torch.equal(bits(x), bits(y)):
                if report:
                    logger.error(f"{report} dev {d}: {what} differs")
                return False
    return True


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_rowtrim_bitwise_and_slope(mesh_device, rows):
    mesh = mesh_device
    mesh.enable_program_cache()
    flaky = 0
    try:
        for step in range(STEPS):
            res, gam, down = make_inputs(mesh, rows, 100 * rows + step)
            off, on, off2 = (run(f, res, gam, down, rows, mesh) for f in (False, True, False))
            if not same(off, off2):
                # a flag-independent flake of the v3 front on this tree (two flag-off runs of the same inputs differ, also in
                # test_gr_front_device and on the tree without this change): the step proves nothing about the flag
                flaky += 1
            elif not same(off, on):
                on = run(
                    True, res, gam, down, rows, mesh
                )  # the flake can hit the flag-on run: it must differ twice to fail
                assert same(off, on, report=f"rows={rows} step {step}"), "flag on differs from flag off"
            for t in (res, gam, down):
                ttnn.deallocate(t)
        logger.info(
            f"rows={rows}: flag off/on bit-identical over {STEPS - flaky} steps ({flaky} baseline-flaky steps skipped)"
        )
        assert flaky <= STEPS // 2, "the baseline flake hides too many steps"
        res, gam, down = make_inputs(mesh, rows, 7)
        times = {"off": [], "on": []}
        for rep in range(3):
            for name, flag in (("off", False), ("on", True)):
                os.environ[front3.ROWTRIM_ENV] = "1" if flag else "0"
                mesh.clear_program_cache()
                times[name].append(traced_us(mesh, lambda: front3.stats_recip_down_gather(res, gam, down)))
                logger.info(f"rows={rows} rep {rep} {name}: slope/call {times[name][-1][0]} us")
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(rows=rows, us=times)) + "\n")
    finally:
        os.environ.pop(front3.ROWTRIM_ENV, None)
