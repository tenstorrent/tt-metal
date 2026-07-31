# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The baseline anchor must not pin a capture the run is about to reject.

before_loop writes the KIND_EAGER anchor at line 704 and only checks the capture at line 716:

    _record_baseline_anchor(profile, model=...)                    # 704 -- PERMANENT, write-once
    ...
    if profile.get("device_ms", 0) <= 0 or _struct_ops == 0:       # 716 -- "refusing to optimize"
        raise RuntimeError("baseline capture looks partial/degenerate ...")

So a degenerate capture is pinned BEFORE anything decides it is degenerate. Anchors are write-once,
so the good measurement that follows the retry lands as an AFTER and can never displace it.

Observed on gemma-3-12b-it, 2026-07-31 10:45. The run printed

    baseline capture looks partial/degenerate (device_ms=0.1004, structural ops=0,
    buckets={'datamove': 26, 'host_overhead': 0}); refusing to optimize against it

and the ledger nonetheless held `eager_per_op before 0.1004 | before_loop`. Every eager "gain vs
baseline" for the rest of that run was computed against 0.1 ms.

perf_mcp's equivalent writer already guards this:

    if phase == led.PHASE_BEFORE and not _is_credible_profile(prof):
        return

_record_baseline_anchor was written by copying that function and dropping that line -- which is why
its 38 tests all passed: every fixture I wrote used a CREDIBLE profile, so the missing guard was
never exercised. The fixture here is the REAL profile file that run left on disk, not one I made up.
"""

import importlib.util
import sys
from pathlib import Path


_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

# The genuine artifact from the 2026-07-31 10:45 run (runs/.../profiles/baseline_profile.json).
_REAL_DEGENERATE = {
    "device_ms": 0.1004,
    "perf_layers": None,
    "buckets": [{"id": "datamove", "count": 26}, {"id": "host_overhead", "count": 0}],
}
_REAL_GOOD = {
    "device_ms": 240.8588,
    "perf_layers": "all",
    "buckets": [{"id": "matmul", "count": 1542, "device_ms": 187.34}],
}


def _led(monkeypatch, path):
    spec = importlib.util.spec_from_file_location("meas_degen", str(_PA / "cc_optimize" / "measurements.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setenv("PERF_MCP_LEDGER", str(path))
    return m


def _bl():
    import agent.before_loop as bl

    return bl


def test_degenerate_capture_is_not_pinned(monkeypatch, tmp_path):
    """THE bug: the real 0.1004 ms capture must leave the BEFORE slot EMPTY."""
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    _bl()._record_baseline_anchor(_REAL_DEGENERATE, model="gemma3", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="gemma3", task="main")
    assert row is None, (
        f"pinned a capture the run rejects as degenerate: {row}. Anchors are write-once, so every "
        "later 'gain vs baseline' is computed against 0.1 ms and cannot be corrected."
    )


def test_the_retry_then_takes_the_before_slot(monkeypatch, tmp_path):
    """The point of refusing: the slot stays free for the credible measurement that follows."""
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    bl._record_baseline_anchor(_REAL_DEGENERATE, model="gemma3", task="main")  # rejected
    bl._record_baseline_anchor(_REAL_GOOD, model="gemma3", task="main")  # the retry
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="gemma3", task="main")
    assert row and float(row["value_ms"]) == 240.8588, f"the real baseline did not become the BEFORE: {row}"


def test_credible_capture_still_pinned(monkeypatch, tmp_path):
    """The guard must not block normal operation."""
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    _bl()._record_baseline_anchor(_REAL_GOOD, model="gemma3", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="gemma3", task="main")
    assert row and float(row["value_ms"]) == 240.8588


def test_an_optimized_model_is_still_credible(monkeypatch, tmp_path):
    """A genuinely fast model must not be mistaken for an empty capture: the run's own optimized
    figure was 40.13 ms, and a later rerun starts from there."""
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    fast = {"device_ms": 40.13, "perf_layers": "all", "buckets": [{"id": "matmul", "count": 1542}]}
    _bl()._record_baseline_anchor(fast, model="gemma3", task="main")
    assert led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="gemma3", task="main")


def test_guard_applies_to_before_only(monkeypatch, tmp_path):
    """AFTER readings are not anchors -- refusing them would silently drop real measurements."""
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    bl._record_baseline_anchor(_REAL_GOOD, model="gemma3", task="main")  # takes BEFORE
    bl._record_baseline_anchor(_REAL_DEGENERATE, model="gemma3", task="main")  # would be an AFTER
    befores = led.rows(led.KIND_EAGER, led.PHASE_BEFORE, "gemma3", "main")
    assert len(befores) == 1 and float(befores[0]["value_ms"]) == 240.8588


def test_it_uses_the_shared_credibility_check_not_a_copy():
    """All three of my ledger bugs were 'copied a writer, dropped a line'. A second, private
    definition of 'credible' would drift from perf_mcp's the moment either changed."""
    src = (_PA / "agent" / "before_loop.py").read_text()
    i = src.index("def _record_baseline_anchor")
    body = src[i : src.index("\ndef ", i + 10)]
    assert "_is_credible_profile" in body, "the anchor does not consult the shared credibility check"


def test_matches_perf_mcp_guard_shape():
    """Side-by-side with the function this was copied from -- the check that was dropped."""
    bl = (_PA / "agent" / "before_loop.py").read_text()
    pm = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    for src, who in ((bl, "before_loop"), (pm, "perf_mcp")):
        assert "PHASE_BEFORE and not _is_credible_profile" in src.replace(
            "led.PHASE_BEFORE", "PHASE_BEFORE"
        ), f"{who} lost the BEFORE-only credibility guard"
