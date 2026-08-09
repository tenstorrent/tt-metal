# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An op's duration is measured WITHIN a core, never across two of them.

tt-perf-report's "Device Time" comes from DEVICE KERNEL DURATION [ns], which process_device_log
computes with `op_first_last`: the FIRST start seen on any core to the LAST end seen on any core.
Where the cores do not share a clock base -- Blackhole -- that span contains the inter-core offset as
well as the op, so the reported "duration" can be the offset rather than the work. The error is
unbounded, and it lands on the number everything downstream is built from: bucket device_ms, the
roofline residual, host_overhead (which is a SUBTRACTION, so an inflated device time drives it
negative and the dispatch row disappears), and the op ranking the optimizer chooses targets from.

tt-metal already computes the right figure and writes it out. The same post-processing runs
`op_core_first_last`, which pairs each core's own start with its own end and never crosses cores, and
emits DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]. Nothing here read it -- so the fix needs no
change to shared profiler code, only that this tool stop reading the wrong column.

MAX rather than AVG: an op is not finished until its SLOWEST core is. Averaging understates every
multi-core op.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent.tracy_tool import _member_device_us, build_buckets  # noqa: E402

_PER_CORE = "DEVICE KERNEL DURATION PER CORE MAX [ns]"


def test_the_per_core_column_wins_over_the_cross_core_one():
    """The case: a cross-core span inflated by an inter-core clock offset, beside the honest
    per-core figure the same capture already contains."""
    m = {"report": {"Device Time": "500000"}, "raw": {_PER_CORE: "600000"}}  # 500 ms vs 0.6 ms
    assert abs(_member_device_us(m) - 600.0) < 1e-6, "read the inflated cross-core value"


def test_it_falls_back_when_the_capture_has_no_per_core_column():
    """Older captures, and hardware whose post-processing does not emit it. A missing column must
    not zero the bucket -- a zero device_ms reads as infinitely fast."""
    assert abs(_member_device_us({"report": {"Device Time": "1234"}, "raw": {}}) - 1234.0) < 1e-6


def test_an_unparsable_or_zero_per_core_value_falls_back_rather_than_zeroing():
    for bad in ("", "0", "n/a", None):
        m = {"report": {"Device Time": "777"}, "raw": {_PER_CORE: bad}}
        assert abs(_member_device_us(m) - 777.0) < 1e-6, bad


def test_the_fastest_core_is_never_taken_for_the_duration():
    """MIN is the fastest core, which is not the op's duration under any reading."""
    from agent.tracy_tool import _PER_CORE_NS_COLS

    assert not any("MIN" in c for c in _PER_CORE_NS_COLS)
    assert any("MAX" in c for c in _PER_CORE_NS_COLS)


def _write(tmp_path, per_core_ns, device_time_us):
    raw = tmp_path / "raw.csv"
    rep = tmp_path / "report.csv"
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, ["GLOBAL CALL COUNT", "ATTRIBUTES", _PER_CORE])
        w.writeheader()
        w.writerow({"GLOBAL CALL COUNT": "1", "ATTRIBUTES": "", _PER_CORE: str(per_core_ns)})
    with open(rep, "w", newline="") as f:
        w = csv.DictWriter(f, ["OP Code", "Global Call Count", "Device Time", "Cores", "Op-to-Op Gap"])
        w.writeheader()
        w.writerow(
            {
                "OP Code": "MatmulDeviceOperation",
                "Global Call Count": "1",
                "Device Time": str(device_time_us),
                "Cores": "64",
                "Op-to-Op Gap": "0",
            }
        )
    return raw, rep


def test_end_to_end_a_bucket_uses_the_per_core_duration(tmp_path):
    """The whole point, through the real bucketiser: an inter-core offset of half a second must not
    become half a second of 'device time' for a 0.6 ms op."""
    raw, rep = _write(tmp_path, per_core_ns=600_000, device_time_us=500_000)
    buckets = build_buckets(rep, raw)
    got = sum(b["device_ms"] for b in buckets)
    assert abs(got - 0.6) < 1e-6, got  # 600000 ns = 0.6 ms, NOT 500 ms
