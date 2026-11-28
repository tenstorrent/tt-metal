# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The depth guard must scope the DEPTH-SENSITIVE key only, never the whole snapshot.

The roofline snapshot mixes two kinds of quantity:

  * ``modeled_floor_ms`` is a SUM OVER THE PROFILED OPS, so it scales with the window it was
    computed at. Rendering a 2-layer floor against a 16-layer measurement is the defect that
    produced the 832.93-vs-1088.15 headline, and the depth guard exists to stop it.
  * ``theoretical_rate`` / ``band`` / ``active_bytes`` / ``bw_fraction`` / ``unit`` are PER-UNIT
    model-level physics -- bytes-per-token over bandwidth. They do not depend on how many layers
    were profiled, so a depth mismatch says nothing about their validity.

The guard discarded the ENTIRE snapshot on a mismatch, which killed the ceiling along with the
floor. Observed on llama3_1_8b_p150: the snapshot was written at ``TT_PERF_LAYERS=16`` and the
report finalized at ``all``, so the computed ceiling (54.577 tok/s/u, band 32.75-43.66,
active_bytes 7,504,924,700) never printed and the report showed ``NO_BAND`` instead.

summary.py reads ``theoretical_rate`` (:482), ``has_unit_ceiling`` (:486) and ``modeled_floor_ms``
(:605) off this snapshot, so nulling it takes out all three.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_CC = Path(__file__).resolve().parent.parent / "cc_optimize"


def _run_module():
    """Load cc_optimize/run.py without importing the package (it pulls in device deps)."""
    sys.path.insert(0, str(_CC.parent))
    spec = importlib.util.spec_from_file_location("cc_run_under_test", str(_CC / "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snapshot(perf_layers):
    """A snapshot shaped exactly like _persist_throughput writes it."""
    return {
        "scope": "model",
        "has_unit_ceiling": True,
        "theoretical_rate": 54.577,
        "band": [32.746, 43.662],
        "active_bytes": 7504924700,
        "peak_bw_gbps": 512.0,
        "tp_degree": 1,
        "bw_fraction": 0.8,
        "bytes_source": "params_rule",
        "unit": "token",
        "modeled_floor_ms": 6.74,
        "perf_layers": perf_layers,
    }


# The depth-invariant keys the report needs to print a ceiling and a band at all.
_CEILING_KEYS = ("has_unit_ceiling", "theoretical_rate", "band", "active_bytes", "bw_fraction", "unit")


def _scope():
    mod = _run_module()
    fn = getattr(mod, "_depth_scoped_throughput", None)
    if fn is None:
        pytest.fail(
            "run.py has no _depth_scoped_throughput helper: the depth guard is still inline at "
            "_emit_summary and sets `_throughput = None`, discarding the depth-INVARIANT ceiling "
            "(theoretical_rate/band/active_bytes) along with the depth-sensitive modeled_floor_ms. "
            "That is what turned llama3_1_8b_p150's 54.577 tok/s/u ceiling into NO_BAND."
        )
    return fn


def test_mismatch_keeps_the_ceiling_and_drops_only_the_floor():
    # Snapshot computed at 16 layers, report finalizing at `all` -- the exact llama3_1_8b_p150 case.
    out = _scope()(_snapshot("16"), "all")
    assert out is not None, "a depth mismatch must not discard the whole snapshot"
    for k in _CEILING_KEYS:
        assert out[k] == _snapshot("16")[k], f"{k} is depth-invariant and must survive a depth mismatch"
    assert out["modeled_floor_ms"] is None, "the floor is a sum over profiled ops and MUST be dropped"


def test_match_keeps_everything_including_the_floor():
    out = _scope()(_snapshot("16"), "16")
    assert out["modeled_floor_ms"] == 6.74, "same depth -> the floor is comparable and must be kept"
    for k in _CEILING_KEYS:
        assert out[k] == _snapshot("16")[k]


def test_unstamped_snapshot_drops_the_floor_but_keeps_the_ceiling():
    # No depth stamp = unknown window (the file predates the stamp). The floor cannot be trusted,
    # but the per-unit ceiling never depended on the window in the first place.
    snap = _snapshot("")
    out = _scope()(snap, "all")
    assert out is not None
    assert out["modeled_floor_ms"] is None
    assert out["theoretical_rate"] == 54.577


def test_none_snapshot_stays_none():
    assert _scope()(None, "all") is None


def test_caller_is_not_mutated():
    """The guard must not corrupt the snapshot dict its caller still holds."""
    snap = _snapshot("16")
    _scope()(snap, "all")
    assert snap["modeled_floor_ms"] == 6.74, "helper must return a copy, not mutate the input"
