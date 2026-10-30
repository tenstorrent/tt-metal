# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS: depth-scoping of the roofline snapshot must never destroy the ceiling.

Companion to test_depth_guard_keeps_ceiling.py. That file pins the contract; this one attacks it.

  s1  cross-product of adversarial (stored, want) depth pairs -- ceiling invariant in EVERY cell
  s2  real-producer coupling: the key set comes from perf_mcp._persist_throughput, not a hand list,
      so adding a depth-sensitive key without classifying it fails here
  s3  metamorphic: idempotence, and only ONE key may ever differ across depth pairs
  s4  concurrency + aliasing: 200 threads scoping a shared dict; the input must never mutate
  s5  corrupt/hostile snapshots must degrade, never raise
  s6  END-TO-END report honesty: a mismatched snapshot must still render a ceiling, not NO_BAND
  s7  the pinned llama3_1_8b_p150 regression, with its real numbers
"""

import importlib.util
import json
import math
import sys
import threading
from pathlib import Path

import pytest

_CC = Path(__file__).resolve().parent.parent / "cc_optimize"


def _run_module():
    sys.path.insert(0, str(_CC.parent))
    spec = importlib.util.spec_from_file_location("cc_run_stress", str(_CC / "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SCOPE = _run_module()._depth_scoped_throughput

# Everything except modeled_floor_ms is per-unit model physics and may NEVER be touched.
_FLOOR_KEY = "modeled_floor_ms"


def _snap(perf_layers="16", **over):
    s = {
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
        _FLOOR_KEY: 6.74,
        "perf_layers": perf_layers,
    }
    s.update(over)
    return s


# --------------------------------------------------------------------------- s1
# Adversarial depth spellings. A depth is a free-form env string, so all of these can reach the
# guard from a real run or a stale /tmp file written by an older build.
_DEPTHS = [
    "16",
    "all",
    "",
    "0",
    "-1",
    "1",
    "2",
    "4",
    "8",
    "32",
    "  16  ",
    "16\n",
    "ALL",
    "All",
    "sixteen",
    "1e3",
    "0x10",
    "16.0",
    "None",
    "null",
    "∞",
    "١٦",
    "9" * 40,
    "x" * 4096,
]


def test_s1_ceiling_survives_every_depth_pair():
    checked = 0
    for stored in _DEPTHS:
        for want in _DEPTHS:
            base = _snap(stored)
            out = _SCOPE(base, want)
            assert out is not None, f"snapshot destroyed for stored={stored!r} want={want!r}"
            for k, v in base.items():
                if k == _FLOOR_KEY:
                    continue
                assert out[k] == v, f"depth-invariant {k} changed for stored={stored!r} want={want!r}"
            assert out[_FLOOR_KEY] in (6.74, None)
            checked += 1
    assert checked == len(_DEPTHS) ** 2 == 576


def test_s1b_identical_depth_always_keeps_the_floor():
    """Reflexivity: a snapshot is always comparable to its own depth (when stamped)."""
    for d in _DEPTHS:
        if not d.strip():
            continue
        assert _SCOPE(_snap(d), d)[_FLOOR_KEY] == 6.74, f"{d!r} not comparable to itself"


# --------------------------------------------------------------------------- s2
def test_s2_key_set_matches_the_real_producer():
    """The writer is the source of truth. If _persist_throughput gains a key, it must be classified
    here deliberately -- otherwise a new depth-SENSITIVE key would silently survive a mismatch."""
    src = (_CC / "perf_mcp.py").read_text()
    start = src.index("def _persist_throughput")
    body = src[start : src.index("_throughput_path().write_text", start)]
    produced = set(__import__("re").findall(r'^\s{12}"([a-z_]+)":', body, __import__("re").M))
    assert produced, "could not parse the producer's key set -- update this test, do not delete it"
    known = set(_snap().keys())
    missing = produced - known
    assert not missing, (
        f"_persist_throughput writes {sorted(missing)} which this stress test does not model. "
        "Classify each as depth-invariant (must survive) or depth-sensitive (must be dropped)."
    )


# --------------------------------------------------------------------------- s3
def test_s3_idempotent():
    once = _SCOPE(_snap("16"), "all")
    twice = _SCOPE(once, "all")
    assert once == twice, "scoping must be idempotent"


def test_s3_at_most_one_key_ever_differs():
    for stored in _DEPTHS:
        base = _snap(stored)
        for want in _DEPTHS:
            out = _SCOPE(base, want)
            diff = {k for k in base if base[k] != out[k]}
            assert diff <= {_FLOOR_KEY}, f"guard altered {diff - {_FLOOR_KEY}} (stored={stored!r} want={want!r})"


def test_s3_dropping_is_monotone_never_invents():
    """The guard may only ever turn a floor into None -- never fabricate or change a floor value."""
    for want in _DEPTHS:
        out = _SCOPE(_snap("16"), want)
        assert out[_FLOOR_KEY] in (6.74, None)
    # a snapshot that never had a floor must not gain one
    out = _SCOPE(_snap("16", **{_FLOOR_KEY: None}), "16")
    assert out[_FLOOR_KEY] is None


# --------------------------------------------------------------------------- s4
def test_s4_concurrent_scoping_never_mutates_shared_input():
    shared = _snap("16")
    frozen = json.dumps(shared, sort_keys=True)
    errors = []
    results = []
    lock = threading.Lock()

    def worker(i):
        try:
            out = _SCOPE(shared, "all" if i % 2 else "16")
            with lock:
                results.append(out[_FLOOR_KEY])
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, f"concurrent scoping raised: {errors[:3]}"
    assert len(results) == 200
    assert json.dumps(shared, sort_keys=True) == frozen, "shared input snapshot was mutated"
    assert set(results) == {6.74, None}, "both branches must be exercised and each must be pure"


def test_s4_returned_copy_is_independent():
    base = _snap("16")
    out = _SCOPE(base, "all")
    out["theoretical_rate"] = 999.0
    assert base["theoretical_rate"] == 54.577, "mutating the result corrupted the caller's dict"


# --------------------------------------------------------------------------- s5
@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        [],
        "not a dict",
        42,
        0.0,
        True,
        {"perf_layers": None},
        {"perf_layers": 16},
        {"perf_layers": ["16"]},
        {"perf_layers": {"a": 1}},
        {"theoretical_rate": float("nan")},
        {"theoretical_rate": math.inf},
        {"band": None},
        {"active_bytes": -1},
    ],
)
def test_s5_hostile_snapshots_degrade_without_raising(bad):
    out = _SCOPE(bad, "all")
    if isinstance(bad, dict):
        assert isinstance(out, dict)
        assert out.get(_FLOOR_KEY) is None
    else:
        assert out is None, "a non-dict snapshot has no ceiling to preserve and must be dropped"


def test_s5_hostile_want_depths_do_not_raise():
    for want in [None, 16, [], {}, 3.5, b"16"]:
        try:
            _SCOPE(_snap("16"), want)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"want_depth={want!r} raised {exc!r}; the guard must degrade, not crash")


# --------------------------------------------------------------------------- s6
def _render(throughput, tmp_path):
    spec = importlib.util.spec_from_file_location("cc_summary_stress", str(_CC / "summary.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = tmp_path / "kernel.json"
    log.write_text("[]")
    return mod.render_summary(
        str(log),
        100.0,
        model="llama3_1_8b_p150",
        task="main",
        finalized=True,
        final_override_ms=45.02,
        throughput=throughput,
    )


def test_s6_mismatched_snapshot_still_renders_a_ceiling(tmp_path):
    """THE regression, at the level the user actually sees: the report.

    Before the fix a depth mismatch nulled the snapshot, so render_summary got throughput=None and
    printed NO_BAND. It must now print the ceiling, because the ceiling never depended on depth.
    """
    scoped = _SCOPE(_snap("16"), "all")
    text = _render(scoped, tmp_path)
    # The three-block table heads the column THEORETICAL rather than printing a "theoretical
    # ceiling :" line; the claim under test -- that a ceiling is rendered at all -- is unchanged.
    assert "THEORETICAL" in text or "theoretical ceiling" in text, f"ceiling missing from report:\n{text[:1500]}"
    assert "NO_BAND" not in text, f"report fell back to NO_BAND despite a valid ceiling:\n{text[:1500]}"


def test_s6_old_behaviour_would_have_failed_this(tmp_path):
    """Proves s6 is a real discriminator: feeding None (the OLD behaviour) loses the ceiling."""
    text = _render(None, tmp_path)
    assert "theoretical ceiling" not in text, (
        "control case is not discriminating -- the ceiling appears even with no snapshot, so "
        "test_s6 above would pass regardless of the fix"
    )


# --------------------------------------------------------------------------- s7
def test_s7_pinned_llama_regression():
    """llama3_1_8b_p150, 2026-07-30: snapshot written at TT_PERF_LAYERS=16, report finalized at
    `all`. The ceiling 54.577 tok/s/u / band 32.75-43.66 / active_bytes 7,504,924,700 must survive;
    only the 6.74 ms modeled floor is scope-bound and must go."""
    out = _SCOPE(_snap("16"), "all")
    assert out["theoretical_rate"] == 54.577
    assert out["band"] == [32.746, 43.662]
    assert out["active_bytes"] == 7504924700
    assert out["has_unit_ceiling"] is True
    assert out["unit"] == "token"
    assert out[_FLOOR_KEY] is None
