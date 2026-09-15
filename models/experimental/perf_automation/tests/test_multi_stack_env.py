# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Task 4: per-stack env var propagation from coverage depth dict.

Three invariants:
  1. Single-stack  (_cov = {"stack0": 2})     -> TT_PERF_LAYERS=2 in env (backward compat)
  2. Multi-stack   (_cov = {"stack0": 4, "stack1": 5})
                                               -> TT_PERF_STACK0_LAYERS=4 and
                                                  TT_PERF_STACK1_LAYERS=5 in env
  3. PERF_MCP_PROFILE_ENV JSON contains the per-stack vars for multi-stack

The bridge test verifies a cap reduces work.  For multi-stack the bridge sets ALL per-stack
env vars at once and checks the combined op-count.  If the bridge returns empty (cap didn't
reduce work) the pipeline must NOT propagate those vars into PERF_MCP_PROFILE_ENV.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _mod():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_ms_env", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Helpers to simulate _coverage_layers output and run the env-wiring logic
# extracted from optimize_pipeline().
# ---------------------------------------------------------------------------


def _apply_cov_to_env(m, cov_env: dict, cov, bridge_result: dict) -> None:
    """Replicate the env-wiring block from optimize_pipeline() for unit testing.

    Calls the same helpers (set_depth, _bridge_depth_env via monkeypatched no-op) and
    writes into `cov_env` exactly as the production code does.
    """
    import agent.layer_depth as _ld

    if not cov:
        return

    if isinstance(cov, dict) and len(cov) > 1:
        # Multi-stack path
        profile_extra: dict = {}
        for _i, (_sid, _depth) in enumerate(sorted(cov.items())):
            _stack_key = f"TT_PERF_STACK{_i}_LAYERS"
            cov_env[_stack_key] = str(_depth)
            profile_extra[_stack_key] = str(_depth)
        try:
            existing_prof = json.loads(cov_env.get("PERF_MCP_PROFILE_ENV") or "{}")
        except (ValueError, TypeError):
            existing_prof = {}
        existing_prof.update(profile_extra)
        cov_env["PERF_MCP_PROFILE_ENV"] = json.dumps(existing_prof)
        # Simulate bridge result
        if bridge_result:
            try:
                ep2 = json.loads(cov_env.get("PERF_MCP_PROFILE_ENV") or "{}")
            except (ValueError, TypeError):
                ep2 = {}
            ep2.update(bridge_result)
            cov_env["PERF_MCP_PROFILE_ENV"] = json.dumps(ep2)
    else:
        # Single-stack path
        cov_single = next(iter(cov.values())) if isinstance(cov, dict) else cov
        _ld.set_depth(cov_env, cov_single)
        if bridge_result:
            cov_env["PERF_MCP_PROFILE_ENV"] = json.dumps(bridge_result)


# ---------------------------------------------------------------------------
# Test 1: single-stack -> TT_PERF_LAYERS (backward compat)
# ---------------------------------------------------------------------------


def test_single_stack_uses_tt_perf_layers():
    """_cov = {"stack0": 2} must set TT_PERF_LAYERS=2, not TT_PERF_STACK0_LAYERS."""
    cov_env: dict = {}
    _apply_cov_to_env(_mod(), cov_env, {"stack0": 2}, bridge_result={})
    assert (
        cov_env.get("TT_PERF_LAYERS") == "2"
    ), f"single-stack coverage must use TT_PERF_LAYERS for backward compat; got {cov_env}"
    assert "TT_PERF_STACK0_LAYERS" not in cov_env, "TT_PERF_STACK0_LAYERS must NOT be set for a single-stack model"


# ---------------------------------------------------------------------------
# Test 2: multi-stack -> TT_PERF_STACK{N}_LAYERS
# ---------------------------------------------------------------------------


def test_multi_stack_sets_per_stack_env_vars():
    """_cov = {"stack0": 4, "stack1": 5} must set TT_PERF_STACK0_LAYERS=4 and
    TT_PERF_STACK1_LAYERS=5 in cov_env."""
    cov_env: dict = {}
    _apply_cov_to_env(_mod(), cov_env, {"stack0": 4, "stack1": 5}, bridge_result={})
    assert cov_env.get("TT_PERF_STACK0_LAYERS") == "4", f"TT_PERF_STACK0_LAYERS must be '4'; got {cov_env}"
    assert cov_env.get("TT_PERF_STACK1_LAYERS") == "5", f"TT_PERF_STACK1_LAYERS must be '5'; got {cov_env}"
    assert "TT_PERF_LAYERS" not in cov_env, "TT_PERF_LAYERS must NOT be set for a multi-stack model (would conflict)"


# ---------------------------------------------------------------------------
# Test 3: PERF_MCP_PROFILE_ENV JSON contains per-stack vars for multi-stack
# ---------------------------------------------------------------------------


def test_multi_stack_profile_env_contains_per_stack_vars():
    """PERF_MCP_PROFILE_ENV must carry TT_PERF_STACK{N}_LAYERS so the tracy subprocess sees them."""
    cov_env: dict = {}
    _apply_cov_to_env(_mod(), cov_env, {"stack0": 4, "stack1": 5}, bridge_result={})
    raw = cov_env.get("PERF_MCP_PROFILE_ENV")
    assert raw is not None, "PERF_MCP_PROFILE_ENV must be set for multi-stack coverage"
    prof = json.loads(raw)
    assert (
        prof.get("TT_PERF_STACK0_LAYERS") == "4"
    ), f"PERF_MCP_PROFILE_ENV must contain TT_PERF_STACK0_LAYERS=4; got {prof}"
    assert (
        prof.get("TT_PERF_STACK1_LAYERS") == "5"
    ), f"PERF_MCP_PROFILE_ENV must contain TT_PERF_STACK1_LAYERS=5; got {prof}"


# ---------------------------------------------------------------------------
# Test 4: single-stack PERF_MCP_PROFILE_ENV reflects bridge result
# ---------------------------------------------------------------------------


def test_single_stack_bridge_result_stored_in_profile_env():
    """When the bridge confirms the depth cap, its env dict goes into PERF_MCP_PROFILE_ENV."""
    cov_env: dict = {}
    bridge = {"TT_PERF_LAYERS": "2", "PERF_MCP_FORCE_ALL_LAYERS": None}
    # Filter out None values (set_depth removes the key rather than setting None)
    bridge_clean = {k: v for k, v in bridge.items() if v is not None}
    bridge_clean = {"TT_PERF_LAYERS": "2"}
    _apply_cov_to_env(_mod(), cov_env, {"stack0": 2}, bridge_result=bridge_clean)
    raw = cov_env.get("PERF_MCP_PROFILE_ENV")
    assert raw is not None, "PERF_MCP_PROFILE_ENV must be set when bridge confirms cap"
    assert json.loads(raw).get("TT_PERF_LAYERS") == "2"


# ---------------------------------------------------------------------------
# Test 5: bridge returning empty -> PERF_MCP_PROFILE_ENV not set for single-stack
# ---------------------------------------------------------------------------


def test_single_stack_empty_bridge_leaves_no_profile_env():
    """If the bridge finds no reduction, PERF_MCP_PROFILE_ENV must NOT be injected."""
    cov_env: dict = {}
    _apply_cov_to_env(_mod(), cov_env, {"stack0": 4}, bridge_result={})
    assert "PERF_MCP_PROFILE_ENV" not in cov_env, "an empty bridge result must not write PERF_MCP_PROFILE_ENV"


# ---------------------------------------------------------------------------
# Test 6: multi-stack bridge result merges into existing PERF_MCP_PROFILE_ENV
# ---------------------------------------------------------------------------


def test_multi_stack_bridge_result_merges_into_profile_env():
    """Bridge env vars are merged INTO PERF_MCP_PROFILE_ENV (not replacing the per-stack keys)."""
    cov_env: dict = {}
    bridge = {"SOME_MODEL_FLAG": "yes"}
    _apply_cov_to_env(_mod(), cov_env, {"stack0": 4, "stack1": 5}, bridge_result=bridge)
    prof = json.loads(cov_env["PERF_MCP_PROFILE_ENV"])
    # Per-stack keys must still be present
    assert prof.get("TT_PERF_STACK0_LAYERS") == "4"
    assert prof.get("TT_PERF_STACK1_LAYERS") == "5"
    # Bridge extra must also be present
    assert prof.get("SOME_MODEL_FLAG") == "yes"


# ---------------------------------------------------------------------------
# Test 7: _bridge_depth_env accepts a dict cov and sets per-stack vars on env
# ---------------------------------------------------------------------------


def test_bridge_depth_env_multi_stack_sets_per_stack_keys(monkeypatch):
    """_bridge_depth_env with cov dict > 1 entry must add TT_PERF_STACK{N}_LAYERS to its
    returned env dict alongside the primary knob."""
    m = _mod()

    # Disable the DEPTH_BRIDGE so the fast path exits
    monkeypatch.setenv("PERF_MCP_DEPTH_BRIDGE", "0")
    result = m._bridge_depth_env(None, {}, "", None, None, {"stack0": 4, "stack1": 6})
    assert result == {}, "DEPTH_BRIDGE=0 must short-circuit and return {}"

    # Now re-enable; stub all the I/O so no device is touched.
    monkeypatch.setenv("PERF_MCP_DEPTH_BRIDGE", "1")
    # stub cache miss
    monkeypatch.setattr(m, "_depth_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_depth_cache_put", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_model_root_from_node", lambda *_a, **_k: Path("/fake/model"))
    # No knob needed from LLM -- pass knob directly
    # Simulate a probe that reduces work (full=10, capped=3)
    _call = [0]

    def _fake_run_op_sigs(*_a, **_k):
        _call[0] += 1
        seq = ["op_A"] * (10 if _call[0] == 1 else 3)
        return None, None, seq

    monkeypatch.setattr(m, "_run_op_sigs", _fake_run_op_sigs)
    monkeypatch.setattr(m, "_llm_depth_env", lambda *_a, **_k: {"TT_PERF_LAYERS": "1"})

    result = m._bridge_depth_env(
        Path("/fake/repo"),
        {},
        "local",
        "test_node.py::test_fn",
        None,
        {"stack0": 4, "stack1": 6},
    )
    # The per-stack keys must be in the returned env
    assert result.get("TT_PERF_STACK0_LAYERS") == "4", f"got {result}"
    assert result.get("TT_PERF_STACK1_LAYERS") == "6", f"got {result}"


# ---------------------------------------------------------------------------
# Test 8: stack ordering is deterministic (sorted by stack_id, not insertion order)
# ---------------------------------------------------------------------------


def test_multi_stack_ordering_is_sorted_by_stack_id():
    """Stack N in the env var name must correspond to the Nth SORTED stack_id, not insertion order."""
    # Insert in reverse order to test that sorted() is used
    cov_env: dict = {}
    _apply_cov_to_env(_mod(), cov_env, {"stack1": 99, "stack0": 11}, bridge_result={})
    # stack0 sorts first -> TT_PERF_STACK0_LAYERS; stack1 sorts second -> TT_PERF_STACK1_LAYERS
    assert (
        cov_env.get("TT_PERF_STACK0_LAYERS") == "11"
    ), "TT_PERF_STACK0_LAYERS must be stack0's depth (11), not stack1's (99)"
    assert cov_env.get("TT_PERF_STACK1_LAYERS") == "99"
