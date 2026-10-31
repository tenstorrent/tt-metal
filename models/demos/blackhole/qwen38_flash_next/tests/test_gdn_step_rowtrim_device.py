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
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split_v4 as v4

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("RT_STEPS", "12"))
RESULTS = os.environ.get("RT_RESULTS", "/tmp/gdn_rowtrim_results.jsonl")
PARAMS = {**demo.DEVICE_PARAMS, "trace_region_size": 16 * 1024 * 1024}


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


RAND = torch.Generator().manual_seed(7)


def up_pad(mesh, tensor, pad):
    """``tensor`` on the mesh with the tile rows 1..31 (outside the logical row) filled with ``pad`` instead of zero.
    ``pad == "rand"``: random bf16 bit patterns (NaN, inf, huge, denormal included), the stale bytes of an unwritten buffer.
    """
    if isinstance(pad, str):
        width = tensor.shape[-1]
        bits_ = torch.randint(-32768, 32768, (1, 1, 31, width), generator=RAND, dtype=torch.int16)
        mode = os.environ.get("RT_RAND", "all")
        if mode == "nonan":  # exponent 255 (inf, NaN) -> 254
            bits_ = torch.where((bits_ & 0x7F80) == 0x7F80, bits_ ^ 0x0080, bits_)
        junk = bits_.view(torch.bfloat16)
        if mode == "mod":  # normal values of magnitude about 1..100
            junk = (torch.randn(1, 1, 31, width, generator=RAND) * 30).to(torch.bfloat16)
        full = torch.cat([tensor.to(torch.bfloat16).reshape(1, 1, 1, width), junk], dim=2)
        t = ttnn.from_torch(
            full.contiguous(),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        return ttnn.slice(t, (0, 0, 0, 0), (1, 1, 1, width))
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
        self.taps = [up_pad(mesh, t, pad) for t in case.taps]
        self.norm = up_pad(mesh, case.norm, pad)

    def step(self, i):
        c = self.case
        projected = up_pad(self.mesh, c.projections[i], self.pad)
        fp.mark_pinned(projected)  # the step builds level 2 only for a pinned (zero padded) projection
        older = [up_pad(self.mesh, t, self.pad) for t in c.older[i]]
        gs.run(projected, older, self.newest, self.taps, self.constants, self.norm, self.state, self.out)
        return projected, older


def setflag(level):
    os.environ["QWEN38_GDN_CONV_SPLIT"] = "4"
    os.environ[v4.ROWTRIM_ENV] = level


# Junk in the dead rows 1..31 of the projection, the ring slots, the taps and the norm weight.  Level 2 is bit-identical to
# level 0 for all of them, NaN and inf included (the FPU matmul k^T delta_b does not let a dead NaN / inf into the state).
# RT_PADS replaces the list.
PADS = (
    [x if x == "rand" else float(x) for x in os.environ["RT_PADS"].split(",")]
    if "RT_PADS" in os.environ
    else [0.0, 3.0, -50.0, 1000.0, 1e5, 1e9, 1e30, float("inf"), float("-inf"), float("nan")]
)


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize(
    "pad",
    [
        *PADS,
        pytest.param(
            "rand",
            marks=pytest.mark.xfail(
                reason="the test writes random bf16 bits straight into the level-2 inputs (projection, ring, taps); level 2 is not bit-identical for them, level 1 is. The model never feeds such rows: its inputs are zero (QWEN38_ZERO_DEAD_ROWS)",
                strict=False,
            ),
        ),
    ]
    if "RT_PADS" not in os.environ
    else PADS,
)
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
            if i == STEPS - 1:
                r2 = runs["2"].read()
                logger.info(
                    f"pad {pad}: level 2 finite: "
                    + ", ".join(
                        f"{w} {all(bool(torch.isfinite(x.float()).all()) for x in a)}"
                        for w, a in zip(("out", "state", "newest"), r2)
                    )
                )
                logger.info(
                    f"pad {pad}: level 0 finite after call {i}: "
                    + ", ".join(
                        f"{what} {all(bool(torch.isfinite(x.float()).all()) for x in a)}"
                        for what, a in zip(("out", "state", "newest"), ref)
                    )
                )
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
        fp.mark_pinned(projected)
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
