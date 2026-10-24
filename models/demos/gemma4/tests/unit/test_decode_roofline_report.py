# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The roofline report reads only the signposted steps and sums the idle gaps as the CSV reports them.
Pure Python (a hand-written Tracy ops CSV)."""

import csv

from models.demos.gemma4.tests.unit import decode_roofline_report

COLS = ["OP CODE", "OP TYPE", "DEVICE ID", "DEVICE FW DURATION [ns]", "OP TO OP LATENCY [ns]"]


def _row(code, op_type="tt_dnn_device", device=0, dur_ns=0, gap_ns=""):
    return dict(zip(COLS, [code, op_type, device, dur_ns, gap_ns]))


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def test_report_keeps_only_the_signposted_steps_and_sums_reported_gaps(tmp_path, capsys, monkeypatch):
    path = tmp_path / "ops_perf_results.csv"
    _write(
        path,
        [
            _row("Matmul", dur_ns=100_000_000, gap_ns=0),  # warmup: outside the signposts
            _row("start", op_type="signpost"),
            _row("Matmul", dur_ns=600_000, gap_ns=100_000),
            _row("AllGatherAsync", dur_ns=400_000, gap_ns=50_000),
            _row("Matmul", op_type="tt_dnn_device", device=1, dur_ns=5_000_000, gap_ns=0),  # another chip
            _row("stop", op_type="signpost"),
            _row("Matmul", dur_ns=100_000_000, gap_ns=0),  # after the steps
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["decode_roofline_report.py", str(path), "--steps", "1", "--params-b", "1", "--chips", "1"],
    )
    decode_roofline_report.main()
    out = capsys.readouterr().out
    assert "rows between signposts (device 0): 2" in out
    assert "device time/step: 1.00 ms  = matmul 0.60 + ccl 0.40 + other 0.00" in out
    assert "op-to-op gaps/step: 0.15 ms" in out
