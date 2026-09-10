"""Unit tests for :func:`compare_hf_tt_probes`.

The comparator pairs HF-side probe records with TT-side probe records
by ``(qualified_name, step)`` and surfaces the FIRST module where any
summary statistic drifts beyond a relative-tolerance threshold. It's
called from the e2e PCC failure path to localize divergence to a
specific module — without raising, so the caller can fall through to
its existing prompt path if anything goes wrong.

These tests pin the contract:

  * Pure decision logic — no I/O, no side effects, never raises.
  * First-in-trace-order divergence wins (not the largest drift).
  * Unpaired modules (probed on one side but not the other) surface
    in the result, not as exceptions.
  * Threshold parameter is honored — same data, different thresholds
    produce different verdicts.
  * Malformed inputs produce empty result with descriptive ``note``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from scripts.tt_hw_planner.agentic.probe import (
    HFModuleStats,
    HFProbeResult,
    compare_hf_tt_probes,
)


def _hf(name: str, step: int, mean: float, std: float, l2: float, abs_max: float) -> HFModuleStats:
    """Build an HFModuleStats with sensible default shape/dtype."""
    return HFModuleStats(
        qualified_name=name,
        class_name="TestModule",
        step=step,
        shape=(1, 4),
        dtype="float32",
        mean=mean,
        std=std,
        l2=l2,
        abs_max=abs_max,
    )


def _tt(name: str, step: int, mean: float, std: float, l2: float, abs_max: float) -> Dict[str, Any]:
    """Build a tt_probe-style record dict matching the HF schema."""
    return {
        "qualified_name": name,
        "class_name": "TestModule",
        "step": step,
        "shape": [1, 4],
        "dtype": "float32",
        "mean": mean,
        "std": std,
        "l2": l2,
        "abs_max": abs_max,
    }


def _hf_result(records: List[HFModuleStats]) -> HFProbeResult:
    return HFProbeResult(
        model_id="test/m",
        records=records,
        num_modules_hooked=len(records),
        decode_steps=list({r.step for r in records}),
        note="ok",
        prompt_text="hello",
        elapsed_s=0.1,
    )


# ─── Empty / malformed inputs ───────────────────────────────────────


def test_empty_records_yield_no_divergence() -> None:
    """Both probes empty → no divergence found, descriptive note."""
    result = compare_hf_tt_probes(_hf_result([]), [])
    assert result.first_divergence is None
    assert result.table == []
    assert result.paired_modules == 0
    assert "no paired modules" in result.note


def test_invalid_hf_input_returns_safe_empty() -> None:
    """Garbage in HF slot → empty result with note, never raises."""
    result = compare_hf_tt_probes("not an HFProbeResult", [])  # type: ignore[arg-type]
    assert result.first_divergence is None
    assert result.paired_modules == 0
    assert "invalid hf_result" in result.note


def test_invalid_tt_input_returns_safe_empty() -> None:
    """Garbage in TT slot → empty result with note, never raises."""
    result = compare_hf_tt_probes(_hf_result([_hf("m", 0, 1, 1, 1, 1)]), "not a list")  # type: ignore[arg-type]
    assert result.first_divergence is None
    assert "invalid tt_records" in result.note


def test_malformed_tt_records_are_silently_skipped() -> None:
    """Mix of valid + garbage TT records → valid records still compared,
    garbage entries don't crash the loop."""
    hf = _hf_result([_hf("a", 0, 1, 1, 1, 1)])
    tt = [
        "not a dict",  # garbage
        {"qualified_name": "a", "step": 0, "mean": 1, "std": 1, "l2": 1, "abs_max": 1},
        {"qualified_name": "a", "step": "bad-step", "mean": 1, "std": 1, "l2": 1, "abs_max": 1},  # bad step type
    ]
    result = compare_hf_tt_probes(hf, tt)  # type: ignore[arg-type]
    assert result.paired_modules == 1
    assert result.note == "ok"


# ─── Threshold semantics ─────────────────────────────────────────────


def test_aligned_records_under_threshold_yield_no_divergence() -> None:
    """HF and TT records identical → no divergence under any threshold.
    With all paired and no orphans, the note is "ok" (genuine clean
    comparison). A different note fires when orphans are present — see
    test_no_divergence_with_orphans_flags_in_note for that path."""
    hf = _hf_result([_hf("layer0", 0, 0.5, 0.1, 1.0, 0.7)])
    tt = [_tt("layer0", 0, 0.5, 0.1, 1.0, 0.7)]
    result = compare_hf_tt_probes(hf, tt, threshold=0.001)
    assert result.first_divergence is None
    assert result.paired_modules == 1
    assert result.note == "ok"


def test_no_divergence_with_orphans_flags_in_note() -> None:
    """Paired modules all within threshold BUT unpaired modules exist
    → note tells caller to inspect orphans. Distinguishes "genuinely
    clean" from "no divergence in what we could compare, but probes
    didn't cover the same set."""
    hf = _hf_result(
        [
            _hf("layer0", 0, 1.0, 0.1, 1.0, 0.7),  # paired
            _hf("layer1", 0, 1.0, 0.1, 1.0, 0.7),  # only-HF
        ]
    )
    tt = [_tt("layer0", 0, 1.0, 0.1, 1.0, 0.7)]
    result = compare_hf_tt_probes(hf, tt, threshold=0.05)
    assert result.first_divergence is None
    assert "unpaired" in result.note


def test_drift_above_threshold_triggers_divergence() -> None:
    """One stat far apart → divergence flagged, contains the right module."""
    hf = _hf_result([_hf("layer0", 0, 1.0, 0.1, 1.0, 0.7)])
    tt = [_tt("layer0", 0, 0.5, 0.1, 1.0, 0.7)]  # mean drifted by 50%
    result = compare_hf_tt_probes(hf, tt, threshold=0.05)
    assert result.first_divergence is not None
    assert result.first_divergence.qualified_name == "layer0"
    assert result.first_divergence.max_drift > 0.05


def test_first_in_trace_order_wins_not_largest_drift() -> None:
    """When multiple modules diverge, return the FIRST in HF-trace order.
    Localization for the iterate loop is most useful at the earliest
    drift point — fixing later ones first wastes effort if upstream is
    already broken."""
    hf = _hf_result(
        [
            _hf("layer0", 0, 1.0, 0.1, 1.0, 0.7),
            _hf("layer1", 0, 1.0, 0.1, 1.0, 0.7),
            _hf("layer2", 0, 1.0, 0.1, 1.0, 0.7),
        ]
    )
    tt = [
        _tt("layer0", 0, 0.9, 0.1, 1.0, 0.7),  # 11% drift on mean
        _tt("layer1", 0, 0.1, 0.1, 1.0, 0.7),  # 90% drift — larger but later
        _tt("layer2", 0, 0.05, 0.1, 1.0, 0.7),  # 95% drift — largest but latest
    ]
    result = compare_hf_tt_probes(hf, tt, threshold=0.05)
    assert result.first_divergence is not None
    assert result.first_divergence.qualified_name == "layer0", "Earliest divergence should win"


def test_threshold_change_changes_verdict() -> None:
    """Same data, tighter threshold catches drift; looser misses it."""
    hf = _hf_result([_hf("layer0", 0, 1.0, 0.1, 1.0, 0.7)])
    tt = [_tt("layer0", 0, 0.95, 0.1, 1.0, 0.7)]  # 5% drift on mean
    assert compare_hf_tt_probes(hf, tt, threshold=0.01).first_divergence is not None  # tight catches it
    assert compare_hf_tt_probes(hf, tt, threshold=0.10).first_divergence is None  # loose misses it


# ─── Unpaired modules ───────────────────────────────────────────────


def test_unpaired_hf_modules_listed_when_tt_missing_records() -> None:
    """HF probed layers TT didn't → flagged in unpaired_hf_modules.
    Typically means TT-side probe didn't hook that module (could be
    a missing graduated component, or probe install missed it)."""
    hf = _hf_result(
        [
            _hf("layer0", 0, 1.0, 0.1, 1.0, 0.7),
            _hf("layer1", 0, 1.0, 0.1, 1.0, 0.7),
        ]
    )
    tt = [_tt("layer0", 0, 1.0, 0.1, 1.0, 0.7)]  # TT didn't probe layer1
    result = compare_hf_tt_probes(hf, tt)
    assert "layer1@0" in result.unpaired_hf_modules
    assert result.paired_modules == 1


def test_unpaired_tt_modules_listed_when_hf_missing_records() -> None:
    """TT probed modules HF didn't (rare; usually means TT ran a path
    HF doesn't, e.g. a fused op exposed as a top-level wrapper)."""
    hf = _hf_result([_hf("layer0", 0, 1.0, 0.1, 1.0, 0.7)])
    tt = [
        _tt("layer0", 0, 1.0, 0.1, 1.0, 0.7),
        _tt("layer_fused", 0, 1.0, 0.1, 1.0, 0.7),  # only on TT
    ]
    result = compare_hf_tt_probes(hf, tt)
    assert "layer_fused@0" in result.unpaired_tt_modules


# ─── Numerical edge cases ───────────────────────────────────────────


def test_zero_zero_yields_zero_drift() -> None:
    """When both sides report a stat as exactly zero, the relative
    drift is zero (not NaN, not inf). Eps in the denominator prevents
    divide-by-zero."""
    hf = _hf_result([_hf("layer0", 0, 0.0, 0.0, 0.0, 0.0)])
    tt = [_tt("layer0", 0, 0.0, 0.0, 0.0, 0.0)]
    result = compare_hf_tt_probes(hf, tt, threshold=0.0001)
    assert result.first_divergence is None
    assert result.paired_modules == 1


def test_nan_treated_as_divergent() -> None:
    """If either side reports NaN, that's a divergence signal — fail
    closed on the suspicious row. Caller sees a module with inf drift
    and can investigate."""
    hf = _hf_result([_hf("layer0", 0, 1.0, 0.1, 1.0, 0.7)])
    tt = [_tt("layer0", 0, float("nan"), 0.1, 1.0, 0.7)]
    result = compare_hf_tt_probes(hf, tt)
    assert result.first_divergence is not None
    assert result.first_divergence.max_drift == float("inf")


def test_missing_stats_skipped_not_crashed() -> None:
    """If a TT record is missing a particular stat field, skip it and
    compute drift over the remaining stats. No crash, no false-OK."""
    hf = _hf_result([_hf("layer0", 0, 1.0, 0.1, 1.0, 0.7)])
    tt_rec = _tt("layer0", 0, 1.0, 0.1, 1.0, 0.7)
    del tt_rec["abs_max"]  # missing one field
    result = compare_hf_tt_probes(hf, [tt_rec])
    assert result.paired_modules == 1  # paired, just on fewer stats
    # No divergence because remaining stats agree
    assert result.first_divergence is None


# ─── Result fields ──────────────────────────────────────────────────


def test_result_carries_full_table_in_hf_trace_order() -> None:
    """The result's table preserves HF-trace order — important for the
    repair-prompt rendering that walks the chain in execution order."""
    hf = _hf_result(
        [
            _hf("c", 0, 1.0, 0.1, 1.0, 0.7),
            _hf("a", 0, 1.0, 0.1, 1.0, 0.7),
            _hf("b", 0, 1.0, 0.1, 1.0, 0.7),
        ]
    )
    tt = [
        _tt("a", 0, 1.0, 0.1, 1.0, 0.7),
        _tt("b", 0, 1.0, 0.1, 1.0, 0.7),
        _tt("c", 0, 1.0, 0.1, 1.0, 0.7),
    ]
    result = compare_hf_tt_probes(hf, tt)
    assert [row.qualified_name for row in result.table] == ["c", "a", "b"]


def test_relative_drift_dict_contains_per_stat_values() -> None:
    """The first-divergence object carries the per-stat drift map so
    a repair prompt can surface WHICH stat moved (mean vs std vs l2)."""
    hf = _hf_result([_hf("layer0", 0, 1.0, 0.1, 1.0, 0.7)])
    tt = [_tt("layer0", 0, 0.5, 0.1, 1.0, 0.7)]
    result = compare_hf_tt_probes(hf, tt, threshold=0.05)
    assert result.first_divergence is not None
    drift_map = result.first_divergence.relative_drift
    assert "mean" in drift_map
    assert drift_map["mean"] > 0.4  # ~0.5 drift relative to max(|1|, |0.5|) = 0.5
    assert drift_map["std"] < 0.01  # std agreed
