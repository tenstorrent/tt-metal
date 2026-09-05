# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The profiling window is the depth the OPS need, not a ceiling nobody chose.

The signpost path computed:

    _cov = min(max(deepest + 1, 2), 16)
                                    ^^

and 16 was the last rung of the 2/4/8/16 ladder that a568d9dcba deleted when it introduced this very
path. Nothing computes a marker capacity anywhere in the tool; the docstring calls 16 "the marker
limit" after the fact. What DOES handle an overflow is profiler_heal + _detect_partial_capture: the
run degrades to a partial report and the capture is FLAGGED as partial, rather than dying on a
TT_FATAL or -- worse -- being read as complete. A depth cap prevents neither.

The cost was silence. On gemma-3-12b-it the window was reported as

    coverage (signposts): 156 distinct op(s) -> TT_PERF_LAYERS=16;
       54 op-type(s) still absent at max depth (present in full model, un-timed)

so a third of the model's op types were outside the timing window -- and the TRUE depth, computed on
the line above as `deepest + 1`, was clamped away before anyone could read it. The log could only
ever say "16"; it could never say "you actually need 19".

Two bounds survive, both real: the model's own declared depth, and PERF_MCP_COV_MAX_DEPTH for a box
that genuinely overflows -- a limit somebody decides, rather than a constant nobody remembers.

  d1  the window is deepest+1, past 16 and well past it
  d2  bounded by the model's declared depth, never beyond it
  d3  PERF_MCP_COV_MAX_DEPTH is an explicit opt-in ceiling
  d4  the floor of 2 holds
  d5  `deep` (ops absent at max depth) tracks the real window, not a stale 16
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_depth", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _run()


# --------------------------------------------------------------------------- d1 NOT CAPPED AT 16
@pytest.mark.parametrize("depth", [17, 19, 24, 32, 48, 80])
def test_d1_depth_beyond_sixteen_is_kept(monkeypatch, depth):
    monkeypatch.delenv("PERF_MCP_COV_MAX_DEPTH", raising=False)
    assert _M._cap_cov_depth(depth) == depth, "the window was clamped to a ceiling nobody chose"


def test_d1_the_gemma3_case(monkeypatch):
    """54 op types first appeared past block 15, so the window must exceed 16."""
    monkeypatch.delenv("PERF_MCP_COV_MAX_DEPTH", raising=False)
    assert _M._cap_cov_depth(19) == 19
    assert _M._cap_cov_depth(19) > 16


def test_d1_the_old_formula_would_have_clamped():
    """Control: what the previous line did to the same input."""
    deepest = 18
    assert min(max(deepest + 1, 2), 16) == 16  # old
    assert _M._cap_cov_depth(max(deepest + 1, 2)) == 19  # new


# --------------------------------------------------------------------------- d2 DECLARED DEPTH
def test_d2_never_deeper_than_the_model(monkeypatch):
    """Profiling deeper than the model exists is nonsense, not caution."""
    monkeypatch.delenv("PERF_MCP_COV_MAX_DEPTH", raising=False)
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: 48)
    assert _M._cap_cov_depth(100, "some/model") == 48
    assert _M._cap_cov_depth(30, "some/model") == 30


def test_d2_unknown_declared_depth_does_not_clamp(monkeypatch):
    monkeypatch.delenv("PERF_MCP_COV_MAX_DEPTH", raising=False)
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: None)
    assert _M._cap_cov_depth(40, "some/model") == 40


# --------------------------------------------------------------------------- d3 EXPLICIT CEILING
def test_d3_env_ceiling_is_honoured(monkeypatch):
    monkeypatch.setenv("PERF_MCP_COV_MAX_DEPTH", "16")
    assert _M._cap_cov_depth(40) == 16


def test_d3_env_ceiling_restores_the_old_behaviour_exactly(monkeypatch):
    """Anyone who needs the old ceiling can have it, deliberately."""
    monkeypatch.setenv("PERF_MCP_COV_MAX_DEPTH", "16")
    for deepest in (1, 5, 15, 18, 47):
        assert _M._cap_cov_depth(max(deepest + 1, 2)) == min(max(deepest + 1, 2), 16)


@pytest.mark.parametrize("junk", ["", "0", "-1", "abc", "16.5"])
def test_d3_junk_ceiling_is_ignored(monkeypatch, junk):
    monkeypatch.setenv("PERF_MCP_COV_MAX_DEPTH", junk)
    assert _M._cap_cov_depth(40) == 40, "a malformed ceiling silently shrank the window"


# --------------------------------------------------------------------------- d4 FLOOR
@pytest.mark.parametrize("depth", [0, 1, 2, -5])
def test_d4_floor_of_two_holds(monkeypatch, depth):
    monkeypatch.delenv("PERF_MCP_COV_MAX_DEPTH", raising=False)
    assert _M._cap_cov_depth(depth) == 2


# --------------------------------------------------------------------------- d5 `deep` TRACKS IT
def test_d5_absent_ops_are_measured_against_the_real_window():
    """`deep` listed ops with first_block >= 16 even when the window was not 16, so it could report
    ops as un-timed that the window actually covers -- or miss ones it does not."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    # anchor on the COVERAGE branch: per_stack_map is populated by _first_block_map
    i = src.index("per_stack_map, _ = _first_block_map(seq)")
    body = src[i : i + 3200]
    assert "b >= _cov" in body, "`deep` still compares against a hardcoded depth"
    assert "b >= 16" not in body, "a stale 16 remains in the absent-ops filter"


def test_d5_no_hardcoded_sixteen_left_in_the_signpost_branch():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("per_stack_map, _ = _first_block_map(seq)")
    body = src[i : i + 3200]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert ", 16)" not in code, "the min(..., 16) clamp is back"
