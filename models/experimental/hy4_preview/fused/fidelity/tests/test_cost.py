# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a pass costs: the time of one fused row matmul with its operands resident in L1 (no DRAM stream, no other
core: unpack + multiply + nothing else), at each pass schedule.

The lab program (lab.py) runs `reps` matmuls in one launch; it is captured as a trace and replayed 20 times back to
back with one synchronize (fused/common/tracing.py), for reps = 1 and reps = 1 + N. (t(1 + N) - t(1)) / N is the time
of one matmul without the launch; t(1) is the launch with one matmul and one pack.

Forms: custom_mm (the tree's, LoFi); "<schedule> merge / finalize / plain" for 8 token rows by how the split
accumulators are handled; "rows16 <schedule>" for 16 token rows (hifi_mm16).

Units:
    q_a           K 6144 (192 K tiles, blocks of 32) x 2 output tiles, BFP8: a projection unit of 64 columns
    q_a_bfp4      the same from BFP4
    gate_up       K 6144 x 2 output tiles (one gate tile, one up tile: a pair of a routed expert), BFP4
    gate_up_piece 48 K tiles x 2 output tiles, BFP4: what one core holds of a pair in the TP 32 split
    down          2 K tiles x 8 output tiles, BFP4, column-major: a down core's columns of one expert
    o_proj        16 K tiles x 4 output tiles, BFP8
    wide          K 6144 x 4 output tiles, BFP8 (the merge's cost grows with the output tiles)

One HY4_MK9 line per (unit, form); all of them also go to $HY4_PROBE_DIR/cost.json. On the mock Galaxy the programs
are compiled only."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.common.tracing import traced_ms
from models.experimental.hy4_preview.fused.fidelity import lab as LAB

TRACE_REGION = 16 << 20
pytestmark = pytest.mark.use_module_device({"trace_region_size": TRACE_REGION})

OUT_DIR = Path(os.environ.get("HY4_PROBE_DIR", "generated/hy4_preview/logs/mk9_probe"))
EXTRA = int(os.environ.get("HY4_FID_COST_REPS", "16"))
CALLS = int(os.environ.get("HY4_FID_COST_CALLS", "8"))
# unit: (kt, ct, format, kb, column-major)
UNITS = {
    "q_a": (192, 2, "bfp8", 32, False),
    "q_a_bfp4": (192, 2, "bfp4", 32, False),
    "gate_up": (192, 2, "bfp4", 48, False),
    "gate_up_piece": (48, 2, "bfp4", 48, False),
    "down": (2, 8, "bfp4", 2, True),
    "o_proj": (16, 4, "bfp8", 16, False),
    "wide": (192, 4, "bfp8", 16, False),
}
# form: (pass schedule, DEST layout of lab.LAYOUTS, the tree's custom_mm): "merge" = split accumulators with the FP32
# merge, "finalize" = with custom_mm's merge, "plain" = no split accumulators
FORMS = {
    "custom_mm": ("lofi", "dense_split", True),
    "lofi finalize": ("lofi", "dense_split", False),
    "lofi merge": ("lofi", "dense_split_exact", False),
    "lofi plain": ("lofi", "dense", False),
    "hifi2 merge": ("hifi2", "dense_split_exact", False),
    "hifi2 plain": ("hifi2", "dense", False),
    "p02 merge": ("p02", "dense_split_exact", False),
    "hifi3 merge": ("hifi3", "dense_split_exact", False),
    "hifi4 merge": ("hifi4", "dense_split_exact", False),
    "hifi4 finalize": ("hifi4", "dense_split", False),
    "hifi4 plain": ("hifi4", "dense", False),
    # 16 token rows (hifi_mm16): the same weights, [16, 32] row tiles
    "rows16 lofi": ("lofi", "dense", False),
    "rows16 hifi2": ("hifi2", "dense", False),
    "rows16 p02": ("p02", "dense", False),
    "rows16 hifi3": ("hifi3", "dense", False),
    "rows16 hifi4": ("hifi4", "dense", False),
}
_units = [u for u in os.environ.get("HY4_FID_COST_UNITS", ",".join(UNITS)).split(",") if u]
_results = []


def timed(device, unit, form, reps):
    kt, ct, fmt, kb, column_major = UNITS[unit]
    fidelity, layout, ref = FORMS[form]
    passes, step = LAB.FIDELITIES[fidelity]
    rows = 16 if form.startswith("rows16") else 8
    op = LAB.Lab(
        device,
        kt,
        ct,
        fmt=fmt,
        rows=rows,
        passes=passes,
        step=step,
        kb=kb,
        reps=reps,
        ref=ref,
        column_major=column_major,
        layout=layout,
    )
    try:
        g = torch.Generator().manual_seed(3)
        op.set_w(torch.randn(LAB.TILE * kt, LAB.TILE * ct, generator=g) * 0.05)
        op.set_x(torch.randn(rows, LAB.TILE * kt, generator=g))
        if LAB.MOCK:
            op()
            return None
        return 1e3 * traced_ms(device, op, repeats=20, rounds=3, calls=CALLS)["replay_ms"]
    finally:
        op.close()


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("unit", _units)
def test_cost(device, unit, form):
    one = timed(device, unit, form, 1)
    many = timed(device, unit, form, 1 + EXTRA)
    if LAB.MOCK:
        return
    kt, ct, fmt, kb, _ = UNITS[unit]
    per = (many - one) / EXTRA
    rows = 16 if form.startswith("rows16") else 8
    mvmuls = 4 * (rows // 8) * kt * ct * LAB.FIDELITIES[FORMS[form][0]][0]
    _results.append(dict(unit=unit, form=form, rows=rows, launch_and_one_us=one, per_matmul_us=per, mvmuls=mvmuls))
    logger.info(
        f"HY4_MK9 cost {unit} ({kt} K tiles x {ct}, {fmt}) [{form}]: {per:.2f} us per matmul "
        f"({1e3 * per / mvmuls:.2f} ns per MVMUL, {mvmuls} MVMULs); the launch with one matmul {one:.2f} us "
        f"({CALLS} calls per trace)"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "cost.json"
    old = json.loads(path.read_text()) if path.exists() else []
    old = [r for r in old if (r["unit"], r["form"]) != (unit, form)]
    path.write_text(json.dumps(old + _results[-1:], indent=1))
