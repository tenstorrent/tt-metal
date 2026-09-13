# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the emit-e2e report must defer to the G6 trace verdict, not the probe.

G6 (trace+1cq capture) is proof the traced pipeline ran on device — a host fallback would break
capture. The categorization probe (`_stub_body_is_native`) can false-flag a graduated, on-device
module: e.g. one that stages an input via `.to()`->from_torch, which the runtime native-probe
miscounts as a torch op. When the trace gate engaged, such a module — graduated on disk (a
`.last_good_native`/`.last_good_sharded` snapshot exists) — is a probe false-positive and must be
CLEARED from the report's CPU-fallback list. This pins that reconciliation, using reliable signals
only (trace-engaged + snapshot existence), never the body-native scan that produced the false flag.

Pure filesystem + list logic — no device, no model/arch value baked in.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.bringup_loop import _safe_id  # noqa: E402
from scripts.tt_hw_planner.commands.emit_e2e import _trace_cleared_fallback  # noqa: E402


def _stub(demo_dir: Path, name: str, *, snapshot: str | None) -> None:
    stubs = demo_dir / "_stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    base = stubs / (_safe_id(name) + ".py")
    base.write_text("# stub body\n")
    if snapshot:
        Path(str(base) + snapshot).write_text("# graduated snapshot\n")


# --------------------------------------------------------------------------- the reconciliation fires
def test_graduated_probe_flag_is_cleared_when_trace_engaged(tmp_path):
    # module graduated on disk (native snapshot) but probe false-flagged it -> trace clears it
    _stub(tmp_path, "hifi_decoder", snapshot=".last_good_native")
    kept, cleared = _trace_cleared_fallback(["hifi_decoder"], tmp_path, trace_is_engaged=True)
    assert kept == [] and cleared == ["hifi_decoder"]


def test_sharded_snapshot_also_counts_as_graduated(tmp_path):
    # the TP-sharded graduation path is a real graduation too -> also cleared
    _stub(tmp_path, "g_p_t", snapshot=".last_good_sharded")
    kept, cleared = _trace_cleared_fallback(["g_p_t"], tmp_path, trace_is_engaged=True)
    assert cleared == ["g_p_t"] and kept == []


# --------------------------------------------------------------------------- the reconciliation stays quiet
def test_no_clear_when_trace_did_not_engage(tmp_path):
    # trace gate did NOT engage -> the report has no proof-of-on-device, keep the probe verdict verbatim
    _stub(tmp_path, "hifi_decoder", snapshot=".last_good_native")
    kept, cleared = _trace_cleared_fallback(["hifi_decoder"], tmp_path, trace_is_engaged=False)
    assert kept == ["hifi_decoder"] and cleared == []


def test_ungraduated_module_is_kept_even_when_trace_engaged(tmp_path):
    # a genuinely-not-graduated module (no snapshot) is a real gap -> trace does NOT clear it
    _stub(tmp_path, "some_missing_op_module", snapshot=None)
    kept, cleared = _trace_cleared_fallback(["some_missing_op_module"], tmp_path, trace_is_engaged=True)
    assert kept == ["some_missing_op_module"] and cleared == []


def test_mixed_list_splits_correctly(tmp_path):
    # graduated ones cleared, ungraduated one kept — the report shows the real remaining gap only
    _stub(tmp_path, "hifi_decoder", snapshot=".last_good_native")
    _stub(tmp_path, "g_p_t", snapshot=".last_good_sharded")
    _stub(tmp_path, "real_gap", snapshot=None)
    kept, cleared = _trace_cleared_fallback(["hifi_decoder", "g_p_t", "real_gap"], tmp_path, trace_is_engaged=True)
    assert kept == ["real_gap"]
    assert sorted(cleared) == ["g_p_t", "hifi_decoder"]


# --------------------------------------------------------------------------- degenerate inputs
def test_empty_fallback_is_noop(tmp_path):
    assert _trace_cleared_fallback([], tmp_path, trace_is_engaged=True) == ([], [])


def test_none_fallback_is_noop(tmp_path):
    assert _trace_cleared_fallback(None, tmp_path, trace_is_engaged=True) == ([], [])


def test_reliable_signals_only_no_body_scan(tmp_path):
    # a graduated module whose body is NOT native-looking (has torch in forward) is STILL cleared —
    # the reconciliation must trust the trace + snapshot, never re-run the buggy body-native scan.
    stubs = tmp_path / "_stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    base = stubs / (_safe_id("hifi_decoder") + ".py")
    base.write_text("def forward(x, g=None):\n    g = g.to(__import__('torch').float32)\n    return g\n")
    Path(str(base) + ".last_good_native").write_text("# snapshot\n")
    kept, cleared = _trace_cleared_fallback(["hifi_decoder"], tmp_path, trace_is_engaged=True)
    assert cleared == ["hifi_decoder"] and kept == []
