# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""summary._measured_stage_ms fed the report's MEASURED stage-timing row from a per-run doc that
only updates when a measurement happens to pass the optional stages_json argument on
record_kernel_attempt. The report's headline full_pipeline_ms lives in a DIFFERENT file (the 1cq
baseline), ratcheted on every real commit. The two update on independent triggers and neither is
authoritative -- observed diverging in both directions on two different models (one run's doc
ahead of its bar, another's behind) -- so the same report could print a correct headline beside a
stale per-stage split with nothing flagging the disagreement.

Fix: prefer the 1cq baseline's own "stages" field per stage, since it is definitionally in sync
with the headline it shares a file with; fall back to the legacy per-run doc only for a stage the
bar has not covered.
"""

import sys
from pathlib import Path

_CC = Path(__file__).resolve().parents[1] / "cc_optimize"
if str(_CC) not in sys.path:
    sys.path.insert(0, str(_CC))

import perf_mcp  # noqa: E402
import summary  # noqa: E402


class _FakeMCP:
    def __init__(self, legacy=None, bar=None):
        self._legacy = legacy or {}
        self._bar = bar or {}

    def read_stage_ms(self, model="", task=""):
        return self._legacy

    def fullpipe_bar_stages(self):
        return self._bar


def test_the_bar_wins_over_the_legacy_doc_for_a_stage_both_cover(monkeypatch):
    monkeypatch.setattr(
        summary,
        "_perf_mcp",
        lambda: _FakeMCP(legacy={"prefill": 599.21, "decode": 599.12}, bar={"prefill": 394.15, "decode": 278.42}),
    )
    assert summary._measured_stage_ms() == {"prefill": 394.15, "decode": 278.42}


def test_the_legacy_doc_fills_in_a_stage_the_bar_has_not_covered(monkeypatch):
    monkeypatch.setattr(
        summary,
        "_perf_mcp",
        lambda: _FakeMCP(legacy={"encode": 15.0, "prefill": 599.21}, bar={"prefill": 394.15}),
    )
    assert summary._measured_stage_ms() == {"encode": 15.0, "prefill": 394.15}


def test_an_empty_bar_falls_back_entirely_to_the_legacy_doc(monkeypatch):
    # Before any 1cq measurement exists at all -- the bar has never been written.
    monkeypatch.setattr(
        summary,
        "_perf_mcp",
        lambda: _FakeMCP(legacy={"prefill": 599.21, "decode": 599.12}, bar={}),
    )
    assert summary._measured_stage_ms() == {"prefill": 599.21, "decode": 599.12}


def test_the_reverse_divergence_is_also_resolved_by_the_bar(monkeypatch):
    """Reproduced on a second model in the opposite direction from the one that found this: the
    legacy doc was NEWER than the bar there. The bar still wins per stage regardless of which side
    happens to be stale -- it is authoritative by construction, not by freshness comparison."""
    monkeypatch.setattr(
        summary,
        "_perf_mcp",
        lambda: _FakeMCP(legacy={"prefill": 408.957, "decode": 11.760}, bar={"prefill": 388.445, "decode": 11.753}),
    )
    assert summary._measured_stage_ms() == {"prefill": 388.445, "decode": 11.753}


def test_no_perf_mcp_reachable_returns_empty(monkeypatch):
    monkeypatch.setattr(summary, "_perf_mcp", lambda: None)
    assert summary._measured_stage_ms() == {}


def test_fullpipe_bar_stages_reads_the_same_file_the_headline_uses(tmp_path, monkeypatch):
    p = tmp_path / "1cq.json"
    p.write_text('{"full_pipeline_ms": 278.42, "stages": {"prefill": 394.15, "decode": 278.42}}')
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", p)
    assert perf_mcp.fullpipe_bar_stages() == {"prefill": 394.15, "decode": 278.42}


def test_fullpipe_bar_stages_drops_a_non_positive_or_non_numeric_entry(tmp_path, monkeypatch):
    p = tmp_path / "1cq.json"
    p.write_text('{"stages": {"prefill": 394.15, "decode": 0, "encode": "n/a"}}')
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", p)
    assert perf_mcp.fullpipe_bar_stages() == {"prefill": 394.15}


def test_fullpipe_bar_stages_fails_closed_on_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "does_not_exist.json")
    assert perf_mcp.fullpipe_bar_stages() == {}


def test_fullpipe_bar_stages_fails_closed_on_a_partial_write(tmp_path, monkeypatch):
    # The live run rewrites this file as new baselines land; a reader that crashes on a mid-write
    # read would take the report down with it.
    p = tmp_path / "1cq.json"
    p.write_text('{"stages": {"prefill": 394.1')
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", p)
    assert perf_mcp.fullpipe_bar_stages() == {}
