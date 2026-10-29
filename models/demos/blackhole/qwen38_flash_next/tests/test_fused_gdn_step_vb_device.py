# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the GDN step split stages on the TP=8 mesh: today's program, stage 1 (=1), stage 2 (=2, decay-first off (default) and on) on identical copies of the same inputs; the gated output, the new state and the newest ring slot must match
today's BIT FOR BIT on every die for STEPS consecutive steps.  Then the traced time per op of today / stage 1 / stage 1
with idle conv cores (QWEN38_GDN_CS_NOWORK=1, the dispatch-cost probe) / stage 2 / stage 2 with decay-first.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0..7 pytest this_file -s."""

import json
import os

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401 (the demo's fixture)
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_gdn_step_cs_device import (
    ENV,
    STEPS,
    Case,
    Run,
    traced_us,
    up,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
RESULTS = os.environ.get("VB_RESULTS", "/tmp/vb_device_results.jsonl")
KNOBS = ("QWEN38_GDN_CS_NOWORK", "QWEN38_GDN_CS_DECAY_FIRST", ENV)
MODES = {  # name: env (None = unset)
    "today": {ENV: "0"},  # stage 1 is the default since x34: today's program needs the explicit 0
    "s1": {ENV: "1"},
    "s2": {ENV: "2"},
    "s2_df": {ENV: "2", "QWEN38_GDN_CS_DECAY_FIRST": "1"},
}
TIMED = {
    **MODES,
    "s1_nowork": {ENV: "1", "QWEN38_GDN_CS_NOWORK": "1"},
    "s2_nowork": {ENV: "2", "QWEN38_GDN_CS_NOWORK": "1"},
}
SKIP_MODES = tuple(m for m in os.environ.get("VB_SKIP", "").split(",") if m)


def setenv(env):
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update(env)


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 16 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_stages_equal_default(mesh_device, rows):
    assert gs.HEADS == 6, "run with MESH_DEVICE=P150x8"
    logger.info(f"load {os.getloadavg()}")
    case = Case(rows, 200 + rows)
    modes = {n: e for n, e in MODES.items() if n not in SKIP_MODES}
    runs = {n: Run(mesh_device, case) for n in modes}
    try:
        for i in range(STEPS):
            keep = []
            for name, env in modes.items():
                setenv(env)
                keep.append(runs[name].step(i))
            ref = runs["today"].read()
            for name in modes:
                if name == "today":
                    continue
                got = runs[name].read()
                for what, a, b in zip(("out", "state", "newest"), ref, got):
                    assert all(torch.equal(x, y) for x, y in zip(a, b)), f"rows={rows} step {i} {name}: {what} differs"
                    assert all(torch.equal(a[0], y) for y in b), f"rows={rows} {name}: dies disagree on {what}"
            for pair in keep:
                for t in (*pair[1], pair[0]):
                    import ttnn

                    ttnn.deallocate(t)
        logger.info(f"rows={rows}: {[n for n in modes if n != 'today']} bit-identical to today over {STEPS} steps")

        projected = up(mesh_device, case.projections[0], gs.BF16)
        older = [up(mesh_device, t, gs.BF16) for t in case.older[0]]
        times = {}
        for rep in range(2):
            for name, env in TIMED.items():
                if name in SKIP_MODES:
                    continue
                setenv(env)
                target = runs.get(name, runs["today"] if name == "today" else runs.get(name.split("_")[0]))
                call = lambda: gs.run(
                    projected,
                    older,
                    target.newest,
                    target.taps,
                    target.constants,
                    target.norm,
                    target.state,
                    target.out,
                )
                call()  # warm the program cache for this variant before the capture
                us = traced_us(mesh_device, call)
                times.setdefault(name, []).append(us[0])
                logger.info(f"rows={rows} rep {rep} {name}: slope {us[0]:.2f} us/op (T12/12 {us[1]:.2f})")
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(rows=rows, steps=STEPS, bitwise=True, us=times, load=os.getloadavg())) + "\n")
    finally:
        setenv({})
