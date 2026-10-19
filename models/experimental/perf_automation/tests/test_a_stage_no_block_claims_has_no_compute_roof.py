# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On a model with several blocks, a stage no block claims gets no compute roof -- not the whole model's.

The memory roof already refused such a stage; the compute term fell through to the model's parameter
count, so Devstral's vision encoder was priced as 24B parameters over its 8192 patches: a 1085 ms floor
under a 34.6 ms measurement, printed as 3135% of peak, and never judged because it sat "below" that
floor. A single-block model is its one block, and keeps the fallback. Stage and block names here are
this file's own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


@pytest.fixture
def summary(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_pinned_peak_flops", lambda *a, **k: 64e12)
    monkeypatch.setattr(S, "_peak_for_stage", lambda *a, **k: (64e12, "hifi4"))
    monkeypatch.setattr(S, "_pinned_ceiling_input", lambda kind, stage, model="", task="": None)
    monkeypatch.setattr(S, "_measured_stage_bytes", lambda model, task: {"alpha": 1e9, "beta": 1e9})
    return S


_GEO = {"layers": 4, "hidden_size": 64, "intermediate_size": 256, "params": 1_000_000}


def _roofs(S, monkeypatch, facts):
    monkeypatch.setattr(S, "_model_facts", lambda: facts)
    return S._stage_roofs(10e9, 288.0, 1, "inference", None, {"alpha": 1.0, "beta": 2.0}, model="m", task="t")


def test_an_unplaced_stage_on_a_two_block_model_has_no_compute_roof(summary, monkeypatch):
    facts = {"total_params": 5_000_000, "blocks": {"ta": dict(_GEO), "tb": dict(_GEO)}, "stage_roots": {"alpha": "ta"}}
    r = _roofs(summary, monkeypatch, facts)
    assert r["alpha"]["compute_ms"] and r["alpha"]["flops"] >= 2 * 1_000_000, "its own block's params"
    assert r["beta"]["compute_ms"] is None and not r["beta"]["flops"], "priced with the whole model's params"
    assert r["beta"]["memory_ms"], "the measured read set still gives it a memory roof"


def test_a_single_block_model_keeps_the_whole_model_fallback(summary, monkeypatch):
    facts = {"total_params": 5_000_000, "blocks": {"ta": dict(_GEO)}}
    r = _roofs(summary, monkeypatch, facts)
    assert r["beta"]["compute_ms"], "one block: the model is that block, as before"


def test_no_blocks_at_all_keeps_the_whole_model_fallback(summary, monkeypatch):
    r = _roofs(summary, monkeypatch, {"total_params": 5_000_000})
    assert r["beta"]["compute_ms"]
