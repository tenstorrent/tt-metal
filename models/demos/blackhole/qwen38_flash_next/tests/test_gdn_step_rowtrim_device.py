# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of QWEN38_GDN_ROWTRIM (gdn_step split 4, rows 1): levels 0 (off), 1 and 2 from identical inputs, STEPS
consecutive calls with the recurrent state carried; the gated output, the carried state and the newest ring slot must be
bit-identical after every call, on every die, for levels 1 and 2 against 0.  Then the traced slope per call, and a zone profile run.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=... pytest this_file -s"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401 (the demo's fixture)
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_gdn_step_cs_device import BF16, Case, Run, traced_us
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split_v4 as v4

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("RT_STEPS", "12"))
RESULTS = os.environ.get("RT_RESULTS", "/tmp/gdn_rowtrim_results.jsonl")
PARAMS = {**demo.DEVICE_PARAMS, "trace_region_size": 16 * 1024 * 1024}


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def up_pad(mesh, tensor, pad):
    """``tensor`` on the mesh with the tile rows 1..31 (outside the logical row) filled with ``pad`` instead of zero."""
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        pad_value=pad,
    )


class PaddedRun(Run):
    """A Run whose projection and ring-slot inputs carry ``pad`` in the dead tile rows (the live row is the same)."""

    def __init__(self, mesh, case, pad):
        super().__init__(mesh, case)
        self.pad = pad

    def step(self, i):
        c = self.case
        projected = up_pad(self.mesh, c.projections[i], self.pad)
        older = [up_pad(self.mesh, t, self.pad) for t in c.older[i]]
        gs.run(projected, older, self.newest, self.taps, self.constants, self.norm, self.state, self.out)
        return projected, older


def setflag(level):
    os.environ["QWEN38_GDN_CONV_SPLIT"] = "4"
    os.environ[v4.ROWTRIM_ENV] = level


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("pad", [0.0, 3.0, -50.0, 1000.0])
def test_rowtrim_bitwise_and_slope(mesh_device, pad):
    """pad = the value of the dead tile rows 1..31 of the inputs (0: what a zero padded tensor has; the others: junk the
    kernels must never let into the live row)."""
    assert gs.HEADS == 6, "run with MESH_DEVICE=P150x8"
    case = Case(1, 301)
    levels = ("0", "1", "2")
    runs = {}
    for level in levels:
        setflag(level)
        runs[level] = PaddedRun(mesh_device, case, pad)
    try:
        for i in range(STEPS):
            keep = []
            for level in levels:
                setflag(level)
                keep.append(runs[level].step(i))
            ref = runs["0"].read()
            for level in levels[1:]:
                for what, a, b in zip(("out", "state", "newest"), ref, runs[level].read()):
                    assert all(
                        torch.equal(bits(x), bits(y)) for x, y in zip(a, b)
                    ), f"call {i} level {level}: {what} differs"
            for pair in keep:
                for t in (*pair[1], pair[0]):
                    ttnn.deallocate(t)
        logger.info(f"levels 1 and 2 bit-identical to 0 over {STEPS} calls (out, state, newest, all dies)")
        projected = up_pad(mesh_device, case.projections[0], pad)
        older = [up_pad(mesh_device, t, pad) for t in case.older[0]]
        times = {level: [] for level in levels}
        for rep in range(3):
            for level in levels:
                setflag(level)
                target = runs[level]
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
                call()
                times[level].append(traced_us(mesh_device, call))
                logger.info(
                    f"rep {rep} level {level}: slope {times[level][-1][0]:.2f} us/call (T12/12 {times[level][-1][1]:.2f})"
                )
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(steps=STEPS, pad=pad, us=times)) + "\n")
    finally:
        os.environ.pop(v4.ROWTRIM_ENV, None)
        os.environ.pop("QWEN38_GDN_CONV_SPLIT", None)


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
def test_zone_profile(mesh_device):
    """Study build (QWEN38_FUSED_ZONES=1 TT_METAL_DEVICE_PROFILER=1, QWEN38_GDN_ROWTRIM = 0 or 1): split-4 gdn_step a few
    times, then the device profiler is read (<profiler dir>/.logs/profile_log_device.csv)."""
    case = Case(1, 301)
    setflag(os.environ.get(v4.ROWTRIM_ENV, "1"))
    run = Run(mesh_device, case)
    try:
        for i in range(4):
            keep = run.step(i)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        for i in range(4, 10):
            keep = run.step(i)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
    finally:
        os.environ.pop(v4.ROWTRIM_ENV, None)
        os.environ.pop("QWEN38_GDN_CONV_SPLIT", None)
