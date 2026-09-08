# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE models route through a data-dependent subset of experts, so two separate probe
invocations of the SAME model at the SAME depth can produce different op-counts -- op-count is not
a reliable "did the cap reduce work" signal for them. block-signpost counts layer *instances*
actually run, which a depth cap always shrinks regardless of which experts got routed to.

Before this fix, _bridge_depth_env only trusted block-signpost when its own ratio already beat the
0.7 threshold, otherwise it silently fell back to op-count -- even when signposts were available and
valid. That meant a model whose signposts showed a real (if narrowly-missed) reduction could still
have its cap rejected -- or accepted -- based purely on which way MoE routing noise happened to
land in that specific pair of probes, observed on nemotron as op-count 38258->39755 (capped > full).

This test builds the exact scenario: valid signposts showing a real depth reduction, alongside an
op-count comparison that MoE noise has made look like no reduction happened at all (capped >= full).
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cc_run_signpost_trust", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_signpost_reduction_is_trusted_despite_moe_op_count_noise(monkeypatch):
    m = _mod()

    monkeypatch.setenv("PERF_MCP_DEPTH_BRIDGE", "1")
    monkeypatch.setattr(m, "_depth_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_depth_cache_put", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_model_root_from_node", lambda *_a, **_k: Path("/fake/model"))
    monkeypatch.setattr(m, "_llm_depth_env", lambda *_a, **_k: {"TT_PERF_LAYERS": "1"})

    _call = [0]

    def _fake_run_op_sigs(*_a, **_k):
        _call[0] += 1
        if _call[0] == 1:
            # "full" probe: 48 block signposts (real depth), but MoE routing keeps op-count LOW.
            seq = [f"{m._SIGNPOST_TOKEN}{i}" for i in range(48)] + ["op_A"] * 100
        else:
            # "capped" probe: only 6 block signposts (real, deep reduction), but MoE routing this
            # time happens to activate more experts, so op-count comes out HIGHER than the full run.
            seq = [f"{m._SIGNPOST_TOKEN}{i}" for i in range(6)] + ["op_A"] * 130
        return None, None, seq

    monkeypatch.setattr(m, "_run_op_sigs", _fake_run_op_sigs)

    result = m._bridge_depth_env(
        Path("/fake/repo"),
        {},
        "local",
        "test_node.py::test_fn",
        None,
        6,
    )

    assert result, (
        "signposts show a real 48->6 depth reduction, but the cap was rejected -- op-count noise "
        "(100->130) is still overriding a valid structural signal"
    )
    assert result.get("TT_PERF_LAYERS") == "6"


def test_op_count_is_still_used_when_no_signposts_exist(monkeypatch):
    """A model with no block signposts at all must still fall back to op-count exactly as before --
    this fix only changes which metric wins when BOTH are available, not the no-signpost path."""
    m = _mod()

    monkeypatch.setenv("PERF_MCP_DEPTH_BRIDGE", "1")
    monkeypatch.setattr(m, "_depth_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_depth_cache_put", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_model_root_from_node", lambda *_a, **_k: Path("/fake/model"))
    monkeypatch.setattr(m, "_llm_depth_env", lambda *_a, **_k: {"TT_PERF_LAYERS": "1"})

    _call = [0]

    def _fake_run_op_sigs(*_a, **_k):
        _call[0] += 1
        seq = ["op_A"] * (100 if _call[0] == 1 else 20)  # no signpost tokens at all
        return None, None, seq

    monkeypatch.setattr(m, "_run_op_sigs", _fake_run_op_sigs)

    result = m._bridge_depth_env(
        Path("/fake/repo"),
        {},
        "local",
        "test_node.py::test_fn",
        None,
        6,
    )

    assert (
        result.get("TT_PERF_LAYERS") == "6"
    ), "a model with no signposts must still be capped via op-count (100->20 is a real reduction)"
