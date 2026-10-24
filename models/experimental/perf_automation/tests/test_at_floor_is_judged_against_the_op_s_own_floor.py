# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An op is at its floor when its gap is small beside ITS floor, never because one profiled pass showed a
small number.

The roofline dropped any op whose gap in the profiling window was under 0.05 ms as "within measurement
noise". The gate weighs a gap by what it costs one request -- a decode row runs every layer on every
step -- but a row dropped here never reached the gate to be weighed. Devstral run 5 (2026-10-19): 0.8
ms per decode step hidden this way, every op with an untried ladder, on a stage 5.85 ms over its band.
Whether a gap is worth a round is the gate's question, asked once per request (_per_request_reader).
Op codes and shapes here are this file's own.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import roofline as R  # noqa: E402

HW = {"arch": "", "dram_bw_gbps": 1000.0, "worker_cores": 100, "peak_tflops_per_core": {"hifi4": 1.0}}


def _op(ms, mb, count=1):
    # 1000 GB/s -> 1 MB takes 0.001 ms, so `mb` megabytes is the memory floor in microseconds.
    return {
        "op_code": "SomeOp",
        "shape": "1x1 @ 1x1",
        "device_ms": ms,
        "count": count,
        "bytes": mb * 1e6,
        "memory": "dram_interleaved",
    }


def _report(*ops):
    # The dispatch floor self-calibrates to the smallest per-call time, so a cheap op sets it low.
    prof = {"device_ms": sum(o["device_ms"] for o in ops), "buckets": [{"id": "other", "top_ops": list(ops)}]}
    return R.residual_report(prof, HW)


def test_a_small_gap_many_times_the_floor_is_open():
    """0.04 ms over a 0.004 ms floor: ten times the floor, under the old absolute cutoff."""
    cheap, op = _op(0.0005, 0.5), _op(0.044, 4.0)
    rep = _report(cheap, op)
    row = next(r for r in rep["rows"] if r["device_ms"] == 0.044)
    assert row["ideal_ms"] == 0.004 and row["gap_ms"] == 0.04
    assert row["at_floor"] is False
    assert any(o["device_ms"] == 0.044 for o in rep["open_ops"])


def test_a_gap_within_a_tenth_of_the_floor_is_at_floor():
    cheap, op = _op(0.0005, 0.5), _op(0.54, 500.0)  # floor 0.5 ms, gap 0.04 = 8%
    row = next(r for r in _report(cheap, op)["rows"] if r["device_ms"] == 0.54)
    assert row["at_floor"] is True


def test_the_classifier_agrees():
    assert R._op_at_floor({"ideal_ms": 0.004, "gap_ms": 0.04}) is False
    assert R._op_at_floor({"ideal_ms": 0.5, "gap_ms": 0.04}) is True


def test_no_absolute_cutoff_remains():
    src = (_PA / "agent" / "roofline.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "_AT_FLOOR_ABS_MS" not in code
