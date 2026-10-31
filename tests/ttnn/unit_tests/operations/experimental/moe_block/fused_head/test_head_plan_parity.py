# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fused head's planner in C++ (ttnn.experimental.moe_block.head_plan_fit) against the reference's answers
(head_plan.fit of mstaletovic/moe-trim, saved by dump_head_plan_golden.py into head_plan_golden.json.gz): the five
models of the block matrix at every rows-a-mesh-row of the matrix on the Galaxy 8 x 4 (line and torus) and the LoudBox
meshes, plus fit()'s keyword variants. Every answer must match exactly: ok, why, note, the ranked layouts with their
predicted time and arena, the first three planned layouts in full (region sizes, the cost's parts) and the
unconstrained pick. Host only, no device:

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/fused_head/test_head_plan_parity.py
"""

import gzip
import json
import math
import time
from pathlib import Path

import pytest

import ttnn

mb = ttnn.experimental.moe_block
GOLDEN = Path(__file__).with_name("head_plan_golden.json.gz")
FULL = 3  # (planned layouts the golden holds in full)


def _cases():
    with gzip.open(GOLDEN, "rt") as f:
        return json.load(f)["cases"]


CASES = _cases()


def _id(c):
    s = c["shape"]
    kw = ",".join(f"{k}={v}" for k, v in sorted(c["kwargs"].items()))
    return (
        f"{c['model']}-{s['R']}x{s['G']}{'-torus' if s['ring'] else ''}-r{s['rows']}-A{s['A']}-L{s['links']}"
        f"-l1{c['l1_free'] // 1024}{'-' + kw if kw else ''}"
    )


def _shape(s):
    return mb.HeadShape(
        rows=s["rows"],
        H=s["H"],
        I=s["I"],
        G=s["G"],
        E=s["E"],
        K=s["K"],
        latent=s["latent"],
        fold=s["fold"],
        links=s["links"],
        ring=s["ring"],
        grid=tuple(s["grid"]),
        R=s["R"],
        sp_ring=s["sp_ring"],
        payload=s["payload"],
        A=s["A"],
        w_dtype={"bf4": mb.HeadWeightDtype.Bf4, "bf8": mb.HeadWeightDtype.Bf8, "bf16": mb.HeadWeightDtype.Bf16}[
            s["w_dtype"]
        ],
    )


def _fit(c):
    kw = dict(c["kwargs"])
    if "placement" in kw:
        kw["placement"] = {"measured": mb.HeadRouterPlacement.Measured, "model": mb.HeadRouterPlacement.Model}[
            kw["placement"]
        ]
    return mb.head_plan_fit(_shape(c["shape"]), l1_free=c["l1_free"], **kw)


def _t(v):
    return "inf" if math.isinf(v) else v


def _lay(l):
    return dict(
        GY=l.GY,
        MT=l.MT,
        NCH=l.NCH,
        KB=l.KB,
        on_engine=l.on_engine,
        GX=l.GX,
        compact=l.compact,
        split=l.split,
        xdram=l.xdram,
    )


def _planned(p):
    if p is None:
        return None
    return dict(lay=_lay(p.lay), arena=p.arena, sizes=dict(p.sizes), t=_t(p.t), parts=dict(p.parts), note=p.note)


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_head_plan_parity(case):
    f = _fit(case)
    ref = case["fit"]
    assert (f.ok, f.why, f.note) == (ref["ok"], ref["why"], ref["note"])
    ranked = [[*_lay(p.lay).values(), _t(p.t), p.arena] for p in f.planned]
    assert ranked == ref["ranked"]
    assert [_planned(p) for p in f.planned[:FULL]] == ref["planned"]
    assert _planned(f.free) == ref["free"]


def test_head_plan_host_time():
    """The planner's host cost (it runs once a block - MoEBlockResources caches it - never per call): the matrix's
    Galaxy cases, us a fit (the reference's Python takes ~0.6 s a fit on these)."""
    cases = [c for c in CASES if c["shape"]["R"] == 8 and not c["kwargs"]]
    t0 = time.perf_counter()
    for c in cases:
        _fit(c)
    us = (time.perf_counter() - t0) / len(cases) * 1e6
    print(f"head_plan_fit: {us:.0f} us a fit over {len(cases)} Galaxy cases")
    assert us < 2e6
