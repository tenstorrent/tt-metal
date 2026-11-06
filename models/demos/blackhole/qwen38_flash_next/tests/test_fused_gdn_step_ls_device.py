# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the lane split GDN step (QWEN38_GDN_LANE_SPLIT=1, rows 8 / 16 / 32) against today's program on the TP=8 mesh.

Two copies of the same inputs (a realistic in-projection row set, three ring slots, four taps, fp32 state, dt/A, norm
weight): today's ``gdn_step.run`` (LANE_SPLIT=0) and the lane split one (=1).  Before EVERY call the output (all 32 tile
rows) is filled with a NaN pattern and the newest ring slot's tile rows with random finite sentinel bytes; the state has
one extra lane (index ``rows``) of random values.  After every step, on every die: out live rows, the new state of the
live lanes and the newest slot's live rows are torch.equal to today's; the dead rows of out and of the newest slot and the
extra state lane are untouched (bit patterns) in both copies; the dies agree.  STEPS (CS_STEPS, 20) consecutive steps (the
state evolves in place; a semaphore left non-zero would show).  A "rand pad rows" case (rows 8 and 16) fills rows
``rows`` .. 31 of the projection and the three older slots with random bf16 bit patterns (NaN, inf, huge, denormal): the
default reads only the lane rows; the lane split must give the same bits.  The timing test prints us/call of both in a
trace (slope of a 4-op against a 12-op trace, as test_fused_gdn_step_cs_device.py).
Every test finishes well inside 600 s for one parameter set: select with -k (go.sh).  VB 2 and 4 are skipped unless
QWEN38_LS_TEST_VB=all.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0..7 pytest this_file -s -k "rows8-vb1".
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401 (the demo's fixture)
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_gdn_step_cs_device import (
    BF16,
    FP32,
    STEPS,
    Case,
    Run,
    per_device,
    traced_us,
    up,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
RESULTS = Path(os.environ.get("LS_RESULTS", "/tmp/ls_device_results.jsonl"))
FLAG, VB, GROUP, HOIST = "QWEN38_GDN_LANE_SPLIT", "QWEN38_GDN_LS_VB", "QWEN38_GDN_LS_G", "QWEN38_GDN_LS_HOIST"
CONVHEAD = "QWEN38_GDN_LS_CONVHEAD"
KNOBS = (FLAG, VB, GROUP, HOIST, CONVHEAD)
ALL_VB = os.environ.get("QWEN38_LS_TEST_VB") == "all"
TILE_ROWS = 32
PARAMS = {**demo.DEVICE_PARAMS, "trace_region_size": 16 * 1024 * 1024}


def vb_param(vb):
    marks = [] if (vb == 1 or ALL_VB) else [pytest.mark.skip(reason="VB 2 / 4: set QWEN38_LS_TEST_VB=all")]
    return pytest.param(vb, id=f"vb{vb}", marks=marks)


VBS = [vb_param(1), vb_param(2), vb_param(4)]
ROWS = [pytest.param(r, id=f"rows{r}") for r in (8, 16, 32)]
HOISTS = [pytest.param(0, id="h0"), pytest.param(1, id="h1")]
# option B: 1 = the conv cores take head items after their conv phase (rows 16 / 32: 96 head cores)
CONVHEADS = [pytest.param(0, id="ch0"), pytest.param(1, id="ch1")]


def setenv(flag, vb=1, group="auto", hoist=0, convhead=0):
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update({FLAG: flag, VB: str(vb), GROUP: str(group), HOIST: str(hoist), CONVHEAD: str(convhead)})


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def same_bits(a, b):
    return torch.equal(bits(a), bits(b))


def random_pad_rows(generator, tensor, rows):
    """``tensor`` [1,1,rows,W] bf16 -> [1,1,32,W] with random bf16 bit patterns in rows ``rows`` .. 31."""
    junk = torch.randint(
        -32768, 32768, (1, 1, TILE_ROWS - rows, tensor.shape[-1]), generator=generator, dtype=torch.int16
    )
    return torch.cat([tensor, junk.view(torch.bfloat16)], dim=2)


def tile_upload(mesh, full32, rows):
    """``full32`` [1,1,32,W] on the mesh, sliced to ``rows`` logical rows (the tile's dead rows keep ``full32``'s bytes,
    the way rowtrim's up_pad does it)."""
    t = up(mesh, full32, BF16)
    return ttnn.slice(t, (0, 0, 0, 0), (1, 1, rows, full32.shape[-1]))


class SRun(Run):
    """A Run with 32-row out / newest tensors (sentinel filled before each step), a state with an extra lane, and
    optional random pad rows in the projection and the older slots."""

    def __init__(self, mesh, case, pad=False):
        super().__init__(mesh, case)
        rows = case.rows
        g = torch.Generator().manual_seed(5000 + rows)
        self.pad = pad
        self.pad_gen = torch.Generator().manual_seed(77 + rows)
        self.sent_out = torch.full((1, 1, TILE_ROWS, gs.VALUE_WIDTH), float("nan"), dtype=torch.bfloat16)
        self.sent_newest = torch.randn(1, 1, TILE_ROWS, gs.QKV_WIDTH, generator=g).to(torch.bfloat16)
        self.extra_lane = torch.randn(1, gs.HEADS, gs.HEAD_DIM, gs.HEAD_DIM, generator=g) * 0.05
        ttnn.deallocate(self.state)
        ttnn.deallocate(self.out)
        ttnn.deallocate(self.newest)
        self.state = up(mesh, torch.cat([case.state, self.extra_lane]), FP32)
        self.out = self.newest = None

    def step(self, i):
        c, rows = self.case, self.case.rows
        for old in (self.out, self.newest):
            if old is not None:
                ttnn.deallocate(old)
        self.out = up(self.mesh, self.sent_out, BF16)
        self.newest = up(self.mesh, self.sent_newest, BF16)
        if self.pad:
            gen = torch.Generator().manual_seed(1000 + i)  # the same junk in both copies
            projected = tile_upload(self.mesh, random_pad_rows(gen, c.projections[i], rows), rows)
            older = [tile_upload(self.mesh, random_pad_rows(gen, t, rows), rows) for t in c.older[i]]
        else:
            projected = up(self.mesh, c.projections[i], BF16)
            older = [up(self.mesh, t, BF16) for t in c.older[i]]
        gs.run(projected, older, self.newest, self.taps, self.constants, self.norm, self.state, self.out)
        return projected, older

    def read(self):
        return per_device(self.out), per_device(self.state), per_device(self.newest)


def compare(rows, case, base, split, base_read, split_read):
    (bo, bs, bn), (so, ss, sn) = base_read, split_read
    live = slice(0, rows)
    dead = slice(rows, TILE_ROWS)
    row = dict(
        out=all(torch.equal(x[..., live, :], y[..., live, :]) for x, y in zip(bo, so)),
        state=all(torch.equal(x[:rows], y[:rows]) for x, y in zip(bs, ss)),
        newest=all(torch.equal(x[..., live, :], y[..., live, :]) for x, y in zip(bn, sn)),
        # informational, not asserted: the split's dead rows of out are bit-equal to the default's dead rows (neither copy
        # keeps the NaN sentinel there, so the sentinel check tested the harness, not the kernels)
        out_dead_same=all(same_bits(x[..., dead, :], y[..., dead, :]) for x, y in zip(bo, so)),
        newest_dead=all(same_bits(x[..., dead, :], base.sent_newest[..., dead, :]) for x in bn)
        and all(same_bits(y[..., dead, :], split.sent_newest[..., dead, :]) for y in sn),
        extra_lane=all(torch.equal(x[rows], base.extra_lane[0]) for x in bs)
        and all(torch.equal(y[rows], split.extra_lane[0]) for y in ss),
        dies_agree=all(torch.equal(bo[0], x) and torch.equal(so[0], y) for x, y in zip(bo, so))
        and all(torch.equal(bs[0], x) and torch.equal(ss[0], y) for x, y in zip(bs, ss)),
        moved=bool((bs[0][:rows] != case.state).any()),
        finite=bool(torch.isfinite(bo[0][..., live, :].float()).all()),
    )
    return row


def assert_ran_lane_split(mesh, ls_run, i, expected_variant):
    """The lane split program must have run (a fall-through to the default would pass the equality trivially): the launch
    record of one extra step carries the variant name."""
    try:
        fp.record_program_meta(True)
        fp.reset_program_meta()
        keep = ls_run.step(i)
        ttnn.synchronize_device(mesh)
        variants = [r.meta.variant for r in fp.program_meta_records()]
        fp.record_program_meta(False)
        for t in (*keep[1], keep[0]):
            ttnn.deallocate(t)
    except Exception as e:  # the check must not fail the numeric test when the recorder is unavailable
        fp.record_program_meta(False)
        logger.warning(f"program record unavailable: {e!r}")
        return
    logger.info(f"launch variants of one ls step: {variants}")
    assert expected_variant in variants, (expected_variant, variants)


def run_case(mesh, rows, vb, group, hoist, pad, label, convhead=0):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import lane_split as ls

    assert gs.HEADS == 6, "run with MESH_DEVICE=P150x8"
    grid = mesh.compute_with_storage_grid_size()
    logger.info(f"{label}: compute grid {grid.x}x{grid.y}; load {os.getloadavg()}")
    plan = ls.plan_for(rows, grid.x, grid.y, vb=vb, g=group, hoist=bool(hoist), convhead=bool(convhead))
    if plan is None:
        if vb != 1:
            pytest.skip(f"no core plan for rows={rows} vb={vb} g={group} on {grid.x}x{grid.y}")
        pytest.fail(f"no core plan for rows={rows} vb={vb} g={group} on {grid.x}x{grid.y}")
    logger.info(
        f"{label}: plan {plan} head cores {plan.head_cores} ({plan.host_cores} of them conv cores) + 40 conv cores"
    )
    case = Case(rows, 300 + rows)
    setenv("0")
    base = SRun(mesh, case, pad)
    setenv("1", vb, group, hoist, convhead)
    split = SRun(mesh, case, pad)
    table = []
    try:
        for i in range(STEPS):
            setenv("0")
            keep_b = base.step(i)
            setenv("1", vb, group, hoist, convhead)
            keep_s = split.step(i)
            row = compare(rows, case, base, split, base.read(), split.read())
            row["step"] = i
            table.append(row)
            for t in (*keep_b[1], keep_b[0], *keep_s[1], keep_s[0]):
                ttnn.deallocate(t)
            if i == 0 and not all(v for k, v in row.items() if k not in ("step", "out_dead_same")):
                break  # no point stepping on
        keys = ("out", "state", "newest", "newest_dead", "extra_lane", "dies_agree", "moved", "finite")
        summary = {k: all(r[k] for r in table) for k in keys}
        logger.info(
            f"{label} bitwise over {len(table)} steps: {summary}; out_dead_same (not asserted) {all(r['out_dead_same'] for r in table)}"
        )
        assert all(summary.values()), [r for r in table if not all(r[k] for k in keys)][:3]
        setenv("1", vb, group, hoist, convhead)
        assert_ran_lane_split(mesh, split, STEPS - 1, plan.variant)
    finally:
        setenv("0")


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("vb", VBS)
@pytest.mark.parametrize("hoist", HOISTS)
@pytest.mark.parametrize("convhead", CONVHEADS)
def test_lane_split_equals_default(mesh_device, rows, vb, hoist, convhead):
    run_case(
        mesh_device, rows, vb, "auto", hoist, False, f"rows={rows} vb={vb} hoist={hoist} convhead={convhead}", convhead
    )


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [pytest.param(8, id="rows8"), pytest.param(16, id="rows16")])
@pytest.mark.parametrize("hoist", HOISTS)
@pytest.mark.parametrize("convhead", CONVHEADS)
def test_lane_split_rand_pad_rows(mesh_device, rows, hoist, convhead):
    run_case(
        mesh_device, rows, 1, "auto", hoist, True, f"rand pad rows={rows} hoist={hoist} convhead={convhead}", convhead
    )


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize(
    "rows,group,hoist,convhead",
    [
        (8, 3, 0, 0),
        (8, 2, 1, 0),
        (16, 4, 1, 0),
        (32, 4, 0, 0),
        (32, 5, 1, 0),
        (32, 8, 1, 0),
        (16, 1, 1, 1),  # option B: 96 head cores (26 conv cores)
        (32, 2, 0, 1),
        (32, 2, 1, 1),
        (32, 3, 1, 1),  # 66 head cores: fits the plain pool, so the option A program runs
    ],
    ids=lambda v: str(v),
)
def test_lane_split_group_variants(mesh_device, rows, group, hoist, convhead):
    """Other lane groupings, uneven last groups included (8 lanes in groups of 3: 3 + 3 + 2; 32 lanes in groups of 5: 6 x 5 + 2)."""
    run_case(
        mesh_device, rows, 1, group, hoist, False, f"rows={rows} g={group} hoist={hoist} convhead={convhead}", convhead
    )


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("hoist", HOISTS)
@pytest.mark.parametrize("convhead", CONVHEADS)
def test_lane_split_timing(mesh_device, rows, hoist, convhead):
    """us per call, today against the lane split, in a trace (the slope of a 4-op against a 12-op trace; two rounds
    each, alternating)."""
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import lane_split as ls

    assert gs.HEADS == 6, "run with MESH_DEVICE=P150x8"
    grid = mesh_device.compute_with_storage_grid_size()
    plan = ls.plan_for(rows, grid.x, grid.y, vb=1, g="auto", hoist=bool(hoist), convhead=bool(convhead))
    assert plan is not None, (rows, grid.x, grid.y)
    case = Case(rows, 400 + rows)
    setenv("0")
    base = SRun(mesh_device, case)
    setenv("1", 1, "auto", hoist, convhead)
    split = SRun(mesh_device, case)
    projected = up(mesh_device, case.projections[0], BF16)
    older = [up(mesh_device, t, BF16) for t in case.older[0]]
    out_b = up(mesh_device, base.sent_out, BF16)
    out_s = up(mesh_device, split.sent_out, BF16)
    newest_b = up(mesh_device, base.sent_newest, BF16)
    newest_s = up(mesh_device, split.sent_newest, BF16)
    times = {}
    try:
        for name, flag, target, out, newest in (
            ("today", "0", base, out_b, newest_b),
            ("ls", "1", split, out_s, newest_s),
            ("today2", "0", base, out_b, newest_b),
            ("ls2", "1", split, out_s, newest_s),
        ):
            setenv(flag, 1, "auto", hoist, convhead)

            def call(target=target, out=out, newest=newest):
                gs.run(projected, older, newest, target.taps, target.constants, target.norm, target.state, out)

            call()  # compile + warm the program cache before the capture
            ttnn.synchronize_device(mesh_device)
            times[name] = traced_us(mesh_device, call)
            logger.info(
                f"rows={rows} hoist={hoist} convhead={convhead} {name}: slope {times[name][0]:.2f} us/call, T12/12 {times[name][1]:.2f} us"
            )
        today = min(times["today"][0], times["today2"][0])
        lane = min(times["ls"][0], times["ls2"][0])
        print(
            f"TIMING rows={rows} vb=1 g={plan.g} hoist={hoist} convhead={convhead} host_cores={plan.host_cores}: today {today:.2f} us/call, lane split {lane:.2f} us/call"
        )
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        rows=rows,
                        vb=1,
                        g=plan.g,
                        hoist=hoist,
                        convhead=convhead,
                        host_cores=plan.host_cores,
                        times=times,
                        load=os.getloadavg(),
                    )
                )
                + "\n"
            )
    finally:
        setenv("0")
