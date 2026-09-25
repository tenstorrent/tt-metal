# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""next_target.reason used to be a per-gate TEMPLATE: identical text on attempt 1 and attempt 5 of
the same op. The exact reason a prior try failed (a crash message, a wedge, a PCC miss) lived only
in the kernel log's own `note` field, and a fresh round starting on that same target had no way to
see it without going and reading the log file itself -- nothing pointed it there or handed it over.

Observed live on the nemotron-3-5-lightning-30b bringup run: the KV-cache target kept re-issuing
after check_full_pipeline_latency errored with no diagnostic content, and the only thing recorded
was "wedged: round killed (UNPRODUCTIVE 10800s ...)" -- a fact a brand-new agent in the next round
had no way to see from next_target alone.

Fix: _last_recorded_note reads the FULL unfiltered log (union of archive + live, same source
_load_attempts_all already uses elsewhere -- see test_tried_is_permanent_not_per_baseline.py) for
the most recent attempt matching this op, and termination_check appends its `note` onto
next_target.reason. Not gated to kv-cache or any other single gate -- it is applied once, at the
one place next_target is assembled, so it carries forward for whichever op/gate is blocking.
"""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _row(sig, kind, note="", **kw):
    r = {"op_signature": sig, "kernel_kind": kind, "note": note}
    r.update(kw)
    return r


def _write_live(mcp, rows):
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps(rows))


def test_the_most_recent_note_for_the_op_is_returned(mcp):
    _write_live(
        mcp,
        [
            _row("generation_loop", "kv-cache", note="reverted: PCC 0.71<0.95"),
            _row("generation_loop", "kv-cache", note="wedged: round killed (UNPRODUCTIVE 10800s)"),
        ],
    )
    assert mcp._last_recorded_note("generation_loop") == "wedged: round killed (UNPRODUCTIVE 10800s)"


def test_a_wedge_row_with_kernel_detected_in_source_false_still_counts(mcp):
    # This is the exact shape _autorecord_wedge writes, and the one the per-op ladder's detected-
    # filtered `attempts` list drops -- _last_recorded_note must read the unfiltered union instead.
    _write_live(
        mcp,
        [_row("generation_loop", "structural-decode", note="wedged: device crash", kernel_detected_in_source=False)],
    )
    assert mcp._last_recorded_note("generation_loop") == "wedged: device crash"


def test_a_different_op_does_not_leak_its_note_onto_this_one(mcp):
    _write_live(mcp, [_row("MatmulDeviceOperation 64 x 2688 x 1856", "grid", note="kept: 4.1->3.6ms")])
    assert mcp._last_recorded_note("generation_loop") == ""


def test_a_different_shape_of_the_same_matmul_class_does_not_match(mcp):
    # _op_match requires the SAME shape when the op carries one -- a note about a different matmul
    # shape must not be handed to an unrelated op just because both are "MatmulDeviceOperation".
    _write_live(mcp, [_row("MatmulDeviceOperation 32 x 32 x 32", "grid", note="reverted: PCC fail")])
    assert mcp._last_recorded_note("MatmulDeviceOperation 64 x 2688 x 1856") == ""


def test_no_attempts_yet_returns_empty(mcp):
    _write_live(mcp, [])
    assert mcp._last_recorded_note("generation_loop") == ""


def test_a_row_with_no_note_is_skipped_in_favour_of_an_older_one_that_has_one(mcp):
    _write_live(
        mcp,
        [
            _row("generation_loop", "kv-cache", note="reverted: PCC 0.71<0.95"),
            _row("generation_loop", "kv-cache", note=""),
        ],
    )
    assert mcp._last_recorded_note("generation_loop") == "reverted: PCC 0.71<0.95"
