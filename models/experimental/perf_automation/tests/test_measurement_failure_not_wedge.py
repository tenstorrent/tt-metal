"""RED tests for BUG 1 + BUG 5 of PERF_AUTOMATION_FIXES_PLAN.md.

BUG 1: a host-side CSV/parse failure is recorded as a DEVICE WEDGE — it burns the
lever, marks a device crash (which triggers a tt-smi reset after two), and renders
in the report as `· wedged` with dashes, reading as "tried and lost on merit".

BUG 5: a profile that yields ZERO op rows is fatal instead of retried. Upstream
tt-metal (`tools/tracy/process_ops_logs.py`) writes a HEADERLESS csv (exactly
b"\\r\\n") and logs success when there are no rows, so the perf tool sees
`unexpected CSV header ... '\\n'`. Reproduced offline 2026-07-25.

Hermetic: no device, no agent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _fresh_perf_mcp(tmp_path):
    """Import perf_mcp against a throwaway manifest (it reads it at import time)."""
    run = tmp_path / "models/experimental/perf_automation/runs/2026-01-01T00-00-00"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps({"config": {"timeout": 10800, "metric": "device_ms"}, "perf_test_resolved": {"path": "t.py"}})
    )
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": 146.72}) + "\n"
    )
    saved = {k: os.environ.get(k) for k in ("PERF_MCP_MANIFEST", "PERF_MCP_KERNEL_LOG")}
    os.environ["PERF_MCP_MANIFEST"] = str(run / "manifest.json")
    os.environ["PERF_MCP_KERNEL_LOG"] = str(tmp_path / "kernlog.json")
    try:
        spec = importlib.util.spec_from_file_location("perf_mcp_under_test", _ROOT / "cc_optimize" / "perf_mcp.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["perf_mcp_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------- BUG 5 upstream repro


def test_headerless_csv_is_what_upstream_writes_for_zero_rows():
    """csv.DictWriter with EMPTY fieldnames writes exactly b'\\r\\n' — read back in text
    mode that is '\\n', which is byte-for-byte the `unexpected CSV header ... '\\n'` seen
    in the llama RUN_REPORT. This is the upstream mechanism, pinned so it cannot be
    mistaken for a device problem again."""
    import csv
    import io

    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=[]).writeheader()
    assert buf.getvalue() == "\r\n"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ops_perf_results_x.csv"
        p.write_bytes(b"\r\n")
        assert p.stat().st_size > 0, "not empty, so it does NOT hit the missing/empty branch"
        assert p.open().readline() == "\n", "text-mode read yields exactly the reported '\\n'"


# --------------------------------------------------------------------------- BUG 5 retry


def test_zero_row_profile_is_retried_not_fatal(tmp_path):
    """A profile that returns no op rows must be RE-PROFILED, not turned into a verdict.
    One retry would have saved all 9 llama attempts on 2026-07-25."""
    m = _fresh_perf_mcp(tmp_path)
    assert hasattr(
        m, "_profile_with_zero_row_retry"
    ), "no retry path exists: a zero-row profile goes straight to the CSV validator and dies"


# --------------------------------------------------------------------------- BUG 1 classification


def test_csv_parse_failure_is_not_classified_as_device_wedge(tmp_path):
    """`unexpected CSV header` is a host-side parse failure. It must NOT be reported as a
    device wedge, must NOT mark a device crash (which resets the board after two), and must
    NOT consume the lever."""
    m = _fresh_perf_mcp(tmp_path)
    assert hasattr(m, "_is_measurement_failure"), "no way to distinguish a parse failure from a device crash"
    assert m._is_measurement_failure("unexpected CSV header in /tmp/x.csv: '\\n'; log: /tmp/y.log")
    assert m._is_measurement_failure("ops CSV missing/empty: /tmp/x.csv")
    assert m._is_measurement_failure("no ops_perf_results_*.csv produced (checked ...)")
    # a real device crash must stay a wedge
    assert not m._is_measurement_failure("Segmentation fault (core dumped)")
    assert not m._is_measurement_failure("TT_FATAL @ tt_cluster.cpp:281")


def test_measurement_failure_verdict_shape(tmp_path):
    """The verdict must say measurement-failed (not REJECTED-as-wedge) and carry a reason
    the agent can act on: 'your edit was NOT measured'."""
    m = _fresh_perf_mcp(tmp_path)
    assert hasattr(m, "_measurement_failed_result"), "no MEASUREMENT_FAILED verdict exists"
    out = m._measurement_failed_result("unexpected CSV header in /tmp/x.csv: '\\n'")
    assert out.get("verdict") == "MEASUREMENT_FAILED"
    assert out.get("measured") is False
    assert "not measured" in (out.get("reason") or "").lower()
    assert out.get("retryable") is True
