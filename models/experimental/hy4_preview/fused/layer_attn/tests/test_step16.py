# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention half's STEP at 16 token rows (P1 + P2 of a shared-indexer layer; PLAN.md section 19.15) and its
alternation with the 8-row step on one arena, as serving switches program sets by the pass's live rows.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k mock
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo and all"
    ... -k "rung and pair_col", "rung and pair_row", "rung and galaxy"        (each after the rung before it has passed)
    ... -k "alternation and solo", "alternation and galaxy", -k latency

A step: the root's two sites (token rows 0..7 and 8..15) on the golden's streams of layer 1, random projection
weights, random stored rows (test_p2.py Case2); P1 then P2 with nothing between them. On a part of the mesh the
absent chips' Xa payloads are host-written before the step (test_p1.py). Checked as test_p2.py checks P2, with the
queries and gates P1 left (every chip's landings; the gate in its pre hub): the attention partials against the torch
partial of the same rows, ctx against the float64 merge of the device partials, the post partial against exact
arithmetic on the stored weights, F against the two-stage sum, every hand-over closed as asked, every word at rest.
Lines start with HY4_MK8A."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.ihc_site.op import golden_pre
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import (
    ALL,
    cb_report,
    galaxy,
    golden_layer,
    site_params,
    streams16,
)
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p2 import (
    LIVE8,
    LIVE16,
    Case2,
    check_p2,
    spread_selection,
    stage_times,
)
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.tests.harness import tables
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.tests.common import hy4_tensors

FID = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
RUNGS = {"solo": [(0, 0)], "pair_col": [(0, 0), (1, 0)], "pair_row": [(0, 0), (0, 1)], "galaxy": None}
MASKS = {"all": 0xFFFF, "nine": 0x01FF, "sparse": 0xA5C3, "one": 0x0100}
FIRST_WAIT = 100_000_000  # polls (about 1 s): an untraced first run launches the chips one after another


class StepCase(Case2):
    """Case2 for a step in which P1 runs too: the queries and the gates are the device's. rows=16: the 16-row layer;
    host=<a 16-row case's layer>: its 8-row program set (token rows 0..7 of the same streams)."""

    def __init__(self, mesh_device, chips=None, **kw):
        kw.setdefault("sink_args", True)
        super().__init__(mesh_device, chips=chips, **kw)
        a = self.a
        self.streams = streams16(golden_layer(1), 0)[: a.rows]
        _, _, cos, sin = tables(16)
        a.set_rope(cos[: a.rows], sin[: a.rows])
        self.launches = 0
        ttnn.synchronize_device(mesh_device)

    def inputs(self):
        """The host's writes before the first step: the absent chips' Xa (a part of the mesh), the streams (16
        rows: the second site's through the arena), the mask."""
        a = self.a
        absent = [c for c in ALL if c not in a.topo.chips]
        if absent:
            (gamma,) = hy4_tensors(["model.layers.1.input_layernorm.weight"])
            y, _, _, _ = golden_pre(a.cfg, *site_params(1), self.streams)
            x = rms_norm(y, gamma.float(), a.cfg.norm_eps).to(torch.bfloat16).float()
            updates, ones = {}, torch.ones(a.rows)
            for coord in absent:
                q_a = L.round7(L.pre_golden(x, ones, self.share(a.topo.chip(coord)))["q_a"])
                for key, items in a.xa_prefill(coord, q_a, L.lane_sums(q_a)).items():
                    updates.setdefault(key, []).extend(items)
            a.write_arena(updates)
        a.set_streams(self.streams)
        if a.mask is not None:
            a.set_mask(self.mask)
        ttnn.synchronize_device(self.mesh)

    def step(self):
        self.a.run_p1()
        self.a.run_p2()
        self.launches += 1

    def queries(self, coord, image):
        """The absorbed queries as P1's X3 landed them on the chip's 8 landing cores: [64 heads, rows, 576]."""
        a, lay = self.a, self.a.lay
        n = 4 * L.HEADS_PER_CHIP * P.LAT_TILES
        groups = []
        for core in A.MERGE:
            raw = A.bytes_bf16(a.arena_bytes(image, core, lay.m_query, n * lay.row_tile), (n, lay.row_tile // 2))
            groups.append(L.from_row_tiles(raw).reshape(a.rows, 8, L.LATENT).permute(1, 0, 2))
        return torch.cat(groups)

    def gate(self, coord, image):
        return self.a.read_p1(coord, image)["gate"]

    def landed(self, coord):
        mine = torch.zeros(64, dtype=torch.bool)
        for r, c in self.a.topo.chips:
            mine[8 * r + 2 * c : 8 * r + 2 * c + 2] = True
        return mine


def l1(mesh_device):
    return ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank


@galaxy
def test_mock(mesh_device):
    """Compiles P1 and P2 at 16 rows and the 8-row program set beside them; L1 by layer and the worst cores."""
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    case16 = StepCase(mesh_device, rows=16)
    a16 = case16.a
    with16 = l1(mesh_device)
    case16.step()
    case8 = StepCase(mesh_device, host=a16, lw=case16.lw)
    a8 = case8.a
    logger.info(
        f"HY4_MK8A step16 mock: persistent L1 per core with the 16-row layer {with16} B, with the 8-row program set beside "
        f"it {l1(mesh_device)} B; arena {a16.lay.arena} B, by role {a16.lay.roles}"
    )
    case8.step()
    for name, a, build, program in (
        ("16-row P1", a16, lambda c: a16._p1_chip(c, False), "p1"),
        ("16-row P2", a16, lambda c: a16._p2_chip(c, "all"), "p2"),
        ("8-row P1", a8, lambda c: a8._p1_chip(c, False), "p1"),
        ("8-row P2", a8, lambda c: a8._p2_chip(c, "all"), "p2"),
    ):
        logger.info(f"HY4_MK8A step16 mock: {name}")
        cb_report(mesh_device, a, build, (("root", (0, 0)), ("other", (0, 1))), program=program)
        for coord in ((0, 0), (0, 1)):
            A.noc_audit(build(coord)[0])
    logger.info("HY4_MK8A step16 mock: the 16-row step and the 8-row step on its layout compiled")


@galaxy
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung, mask):
    """The 16-row step on one chip, two chips, all 32: twice on the same inputs, the second time with nothing
    written by the host (the words and the arena as the first step left them). On a part of the mesh the arena image
    is written again before the second step: P2 leaves an attention core's partial where the query hub holds the
    ABSENT chips' host-written q_a tiles (PLAN.md section 4, the arena's constraint), which only a real Xa rewrites."""
    case = StepCase(mesh_device, chips=RUNGS[rung], rows=16, mask=MASKS[mask], wait_limit=FIRST_WAIT)
    case.inputs()
    for step in (1, 2):
        if step == 2 and RUNGS[rung] is not None:
            case.a.write_arena({})
        case.step()
        if A.is_mock():
            return
        ttnn.synchronize_device(mesh_device)
        check_p2(case, "all", f"step16 {rung} mask {mask} step {step}")


@galaxy
@pytest.mark.parametrize("rung", ["solo", "galaxy"])
def test_scribble(mesh_device, rung):
    """The second site's rows are ranges of the root's ARENA, which the model keeps out of its restore image as
    scratch: "every range is written by a step before the step reads it" (PLAN.md 19.15). The proof for those
    ranges: a step; then random bytes over all of them (streams, y, x, coefficient page, statistic page: what a pass
    of other programs between two steps may leave), the streams written again as the step's first program does (the
    embedding; here set_streams), and the same step again: F must be the first step's bit for bit (fixed merge
    order), every check of the step passing."""
    case = StepCase(mesh_device, chips=RUNGS[rung], rows=16, mask=0xFFFF, fixed_order=True, wait_limit=FIRST_WAIT)
    a = case.a
    case.inputs()
    case.step()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    check_p2(case, "all", f"step16 scribble {rung}: before")
    first = read_f(case)
    a.scribble_site2(seed=17)
    image = a.read_arena(a.topo.root)
    lay = a.lay
    c = a.site2.cores[0]
    noise = a.arena_bytes(image, (c.x, c.y), lay.s2_y, lay.s2_end - lay.s2_y)
    assert len(set(noise)) > 200, "the scribble is on the device"
    a.set_streams(case.streams)
    ttnn.synchronize_device(mesh_device)
    case.step()
    ttnn.synchronize_device(mesh_device)
    check_p2(case, "all", f"step16 scribble {rung}: after")
    d = differ(read_f(case), first)
    logger.info(
        f"HY4_MK8A step16 scribble {rung}: random bytes over the second site's arena ranges ({lay.s2_end - lay.s2_x} B on "
        f"each of its {len(a.site2.cores)} cores) between two steps, the streams written again: {d} of {first.numel()} "
        f"elements of F differ from the step before"
    )
    assert d == 0, f"{d} elements of F changed after the second site's ranges were scribbled"


def read_f(case):
    return case.a.read_root()["f"].clone()


def differ(x, y):
    return int((x != y).sum())


@galaxy
@pytest.mark.parametrize("rung", ["solo", "galaxy"])
def test_alternation(mesh_device, rung):
    """The 8-row step and the 16-row step in turn on one arena, words and site, as serving alternates them, in the
    configuration of record (fixed merge order: a step's result does not depend on arrival order, so "as alone" is
    bit for bit). The host writes the inputs ONCE, before the first step; after that nothing but launches.
      16, 16          the 16-row step on the layout as built: its F twice (the reference of "alone")
      8, 16, 8, 16, 8 in turn: every 16-row F equal to the reference, every 8-row F equal to the first; words at
                      rest and hand-overs closed after each
      trace           [8-row step, 16-row step] as ONE trace, replayed: both F after every replay equal to theirs
    then, with both released, the 8-row step on a layer of its own (same weights, rows, streams): F equal to the
    8-row steps' on the shared layout."""
    chips = RUNGS[rung]
    case16 = StepCase(mesh_device, chips=chips, rows=16, mask=0xFFFF, fixed_order=True, wait_limit=FIRST_WAIT)
    case8 = StepCase(mesh_device, chips=chips, host=case16.a, lw=case16.lw, fixed_order=True, wait_limit=FIRST_WAIT)
    a8, a16 = case8.a, case16.a
    assert a8.arena is a16.arena and a8.words is a16.words and a8.site is a16.site and a8.streams is a16.streams
    case16.inputs()  # (the 8-row set reads token rows 0..7 of the same streams tensor)
    if chips is not None:
        case8.inputs()
    f16, f8 = [], []

    def run(case, sink, label):
        if chips is not None:
            # a part of the mesh (bring-up): the absent chips' Xa is host-written, and the two sets' tiles of it share
            # the query hub's buffer, so each step's arena image is written again before it. On 32 chips: nothing.
            case.a.write_arena({})
        case.step()
        if A.is_mock():
            return
        ttnn.synchronize_device(mesh_device)
        check_p2(case, "all", f"alternation {rung} {label}")
        sink.append(read_f(case))

    run(case16, f16, "16 rows, first")
    run(case16, f16, "16 rows, again")
    for turn in range(3):
        run(case8, f8, f"turn {turn}: 8 rows")
        if turn < 2:
            run(case16, f16, f"turn {turn}: 16 rows")
    if A.is_mock():
        return
    d16 = [differ(f, f16[0]) for f in f16[1:]]
    d8 = [differ(f, f8[0]) for f in f8[1:]]
    logger.info(
        f"HY4_MK8A alternation {rung}: elements of F that differ from the first run of the same step: 16-row steps "
        f"(again, turn 0, turn 1) {d16} of {f16[0].numel()}; 8-row steps (turns 1, 2) {d8} of {f8[0].numel()}"
    )

    def pair():
        a8.run_p1()
        a8.run_p2()
        a16.run_p1()
        a16.run_p2()

    replays = int(os.environ.get("HY4_ATTN_REPLAYS", "20")) if chips is None else 0
    t8, t16 = [], []
    trace_id = None
    if replays:
        # (a capture records the programs and launches nothing: the senders' counts of closed hand-overs say so)
        trace_id, _ = tracing.capture(mesh_device, pair)
    try:
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            case8.launches += 1
            case16.launches += 1
            t8.append(differ(read_f(case8), f8[0]))
            t16.append(differ(read_f(case16), f16[0]))
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
    status = {**a8.status(), **a16.status()}
    logger.info(
        f"HY4_MK8A alternation {rung}: [8-row step, 16-row step] as one trace, {replays} replays: replays whose 8-row F "
        f"differs from the untraced one {sum(1 for d in t8 if d)}, whose 16-row F differs {sum(1 for d in t16 if d)}; "
        f"words after the replays: {status or 'at rest'}"
    )
    if replays:
        check_p2(case16, "all", f"alternation {rung} after the traced replays: 16 rows")
    for case in (case8, case16):
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.sel)
    a8.release()
    a16.release()
    own = StepCase(mesh_device, chips=chips, lw=case16.lw, fixed_order=True, wait_limit=FIRST_WAIT)
    own.inputs()
    own.step()
    ttnn.synchronize_device(mesh_device)
    check_p2(own, "all", f"alternation {rung}: the 8-row step on a layer of its own")
    alone = differ(read_f(own), f8[0])
    logger.info(
        f"HY4_MK8A alternation {rung}: the 8-row step on a layer of its own against the shared layout's: {alone} of {f8[0].numel()} elements of F differ"
    )
    ttnn.deallocate(own.kv)
    ttnn.deallocate(own.sel)
    own.a.release()
    assert not any(d16) and not any(d8), f"a step's F changed between turns: 16 rows {d16}, 8 rows {d8}"
    assert not any(t8) and not any(t16), f"a traced replay's F differs: 8 rows {t8}, 16 rows {t16}"
    assert not status, f"words not at rest after the traced replays: {status}"
    assert (
        alone == 0
    ), f"the 8-row step on the shared layout differs from the step on a layer of its own in {alone} elements"


@galaxy
def test_latency(mesh_device):
    """P1 + P2 as one trace on 32 chips, 8 steps per trace: the 16-row step against the number of live rows, and the
    8-row step as the 16-row layer's second program set (what serving runs for 8 live rows or fewer) in the same
    job; every live row's 2,048 picks spread at random over the chips. HY4_ATTN_TIMING=1: with stamps, and P2's
    stages of the last replay."""
    timing = bool(os.environ.get("HY4_ATTN_TIMING"))
    out = {}
    case16 = StepCase(mesh_device, rows=16, timing=timing, keep_out=False, counts=[64] * 16)
    case8 = StepCase(mesh_device, host=case16.a, lw=case16.lw, timing=timing, keep_out=False, counts=[64] * 8)
    case16.inputs()
    for case, masks in ((case16, LIVE16), (case8, LIVE8)):
        a, rows = case.a, case.a.rows
        for name, live in masks.items():
            lo, hi, mean = spread_selection(case, 0xFFFF if rows != L.ROWS else live)
            if rows != L.ROWS:
                case.mask = live
                a.set_mask(live)
            ttnn.synchronize_device(mesh_device)
            case.step()
            if A.is_mock():
                continue
            ttnn.synchronize_device(mesh_device)
            status = a.status()
            assert not status, f"{rows} rows, {name} live: words not at rest after a step: {status}"
            r = tracing.traced_ms(mesh_device, case.step, calls=8)
            out[(rows, name)] = r["replay_ms"] * 1e3
            n = int(name.split()[0])
            logger.info(
                f"HY4_MK8A step16 latency [{FID}{', fixed order' if a.fixed_order else ''}, timing={timing}] {rows}-row step, {name} "
                f"live row(s): {r['replay_ms'] * 1e3:.1f} us per step, 8 steps per trace (rounds {[round(v * 1e3, 1) for v in r['rounds_ms']]}) "
                f"= {r['replay_ms'] * 1e3 / n:.1f} us per live row, {n / (78 * r['replay_ms'] * 1e-3):.0f} tokens/s if a pass were "
                f"78 of this half alone; selected rows per chip and live row {lo} to {hi}, mean {mean:.1f}"
            )
            status = a.status()
            assert not status, f"{rows} rows, {name} live: words not at rest after the traces: {status}"
            if timing:
                stage_times(case, f"step16 latency [{FID}] {rows}-row step, {name} live, P2")
    for case in (case8, case16):
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.sel)
    case8.a.release()
    case16.a.release()
