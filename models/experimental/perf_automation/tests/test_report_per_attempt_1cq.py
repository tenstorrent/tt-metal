# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The per-attempt table must show the metric the WIN is judged on.

The `✓ win` is decided on trace+1CQ end-to-end, but the delta column beside it printed an EAGER
device_ms delta (baseline − measured) — a different ruler — so a real win could show a delta that
didn't correspond to it. This pins the corrected table:

  * headings trimmed to `Per-attempt detail:` / `Code changes:`
  * the raw column is labelled `Eager (device_ms)` (kept as the per-op steering signal)
  * the delta column is `1CQ Δ vs current` = fullpipe_ms − fullpipe_best_ms (this attempt's 1CQ vs the
    running 1CQ baseline it was measured against): `−` = time went DOWN (faster), `+` = up.
  * an attempt with no 1CQ reading renders `—`, never a fabricated number.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _S():
    spec = importlib.util.spec_from_file_location("summary_1cq_ut", _ROOT / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _render(attempts):
    kl = Path(tempfile.mkdtemp()) / "kl.json"
    kl.write_text(json.dumps(attempts))
    out = _S().render_summary(str(kl), model="m", task="main", metric="device_ms", baseline_ms=11.50, finalized=True)
    return out if isinstance(out, str) else "\n".join(out)


_ATTEMPTS = [
    {  # a win: 1CQ dropped 11.50 -> 10.60
        "op_signature": "MatmulDeviceOperation",
        "kernel_kind": "fidelity",
        "measured_ms": 184.5,
        "fullpipe_ms": 10.60,
        "fullpipe_best_ms": 11.50,
        "beat_baseline": True,
        "diff": "- a\n+ b",
        "note": "lofi",
    },
    {  # no gain: 1CQ went UP 11.50 -> 11.51
        "op_signature": "AllGatherDeviceOperation",
        "kernel_kind": "grid",
        "measured_ms": 184.4,
        "fullpipe_ms": 11.51,
        "fullpipe_best_ms": 11.50,
        "beat_baseline": False,
        "note": "up",
    },
    {  # no 1CQ reading at all
        "op_signature": "SliceDeviceOperation",
        "kernel_kind": "shard",
        "measured_ms": 180.0,
        "beat_baseline": False,
        "note": "no fp",
    },
]


def test_headings_are_trimmed():
    out = _render(_ATTEMPTS)
    assert "Per-attempt detail:" in out
    # the old verbose headings must be gone
    assert "every optimization tried — win OR fail" not in out
    assert "Code changes — every attempt" not in out


def test_the_report_carries_no_source_diffs():
    """The Code changes section printed the full patch of EVERY attempt, win or fail -- thousands of
    lines of diff in a document read for its numbers. The diffs live in the kernel log and in git."""
    out = _render(_ATTEMPTS)
    assert "Code changes:" not in out, out
    assert not [ln for ln in out.splitlines() if ln.startswith("[#")], out


def test_the_attempt_table_carries_no_prose_column():
    """It held the agent's own reasoning, truncated at 200 characters, so every row ended mid-sentence
    and the four measured columns were crowded out by text that was never a measurement."""
    out = _render(_ATTEMPTS)
    assert "why tried" not in out, out


def test_columns_are_eager_and_1cq():
    out = _render(_ATTEMPTS)
    assert "eager device_ms" in out
    assert "1CQ Δ vs current" in out
    assert "gain vs base" not in out  # the old eager-masquerading label is gone


def _detail_rows(out):
    """Only the per-attempt table rows (a 'Matmul' also appears in the coverage grid above)."""
    lines = out.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("Per-attempt detail:"))
    j = next((n for n, ln in enumerate(lines) if n > i and ln.startswith("Limitations")), len(lines))
    return lines[i:j]


def test_delta_is_1cq_vs_current_not_eager():
    rows = _detail_rows(_render(_ATTEMPTS))
    mm = next(ln for ln in rows if ln.lstrip().startswith("Matmul"))
    ag = next(ln for ln in rows if ln.lstrip().startswith("AllGather"))
    sl = next(ln for ln in rows if ln.lstrip().startswith("Slice"))
    # win: 10.60 - 11.50 = -0.90 (faster). NOT the eager delta (baseline 11.50 - 184.5).
    assert "-0.90 ms" in mm and "184.50" in mm
    assert "+173" not in mm and "-173" not in mm  # would be the old eager delta
    # no-gain: 11.51 - 11.50 = +0.01 (slower)
    assert "+0.01 ms" in ag
    # no 1CQ reading -> em dash, never fabricated
    assert "—" in sl


def test_eager_column_still_carries_device_ms():
    # the raw per-op device_ms (steering signal) is retained, just relabelled
    mm = next(ln for ln in _detail_rows(_render(_ATTEMPTS)) if ln.lstrip().startswith("Matmul"))
    assert "184.50" in mm
