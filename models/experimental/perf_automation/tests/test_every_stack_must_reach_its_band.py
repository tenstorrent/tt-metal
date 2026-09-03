"""A run is not done while any stack sits above its own achievable band.

Stopping was judged against the headline, and the headline is ONE stage. voxtral 2026-09-03 finished
with the recurring stage at 10.99 ms against a 21-28 ms band -- past it, which is fine -- while the
prompt stage sat at 182.44 against 26-35 and the audio stage at 38.49 against its own. can_stop was
decided without either being looked at.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (PERF, PERF / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = _ilu.spec_from_file_location("_pm_band", PERF / "cc_optimize" / "perf_mcp.py")
_pm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pm)


class _Roofs:
    """Stand in for summary._stage_roofs with a fixed answer."""

    def __init__(self, roofs):
        self._roofs = roofs

    def _stage_roofs(self, *a, **k):
        return self._roofs


def _arrange(monkeypatch, roofs, measured, band=(60.0, 80.0), theo=100.0):
    monkeypatch.setattr(
        _pm,
        "_read_throughput",
        lambda: {
            "active_bytes": 1_000_000,
            "peak_bw_gbps": 512.0,
            "tp_degree": 1,
            "unit": "token",
            "band": list(band),
            "theoretical_rate": theo,
        },
    )
    # The bytes and the unit come from their OWNERS, not off the throughput snapshot -- see the
    # single-source-of-truth registry. Patch them where the helper actually reads them.
    monkeypatch.setattr(_pm, "_load_perf_target_inputs", lambda *a, **k: {"unit": "token"})
    monkeypatch.setattr(_pm, "_anchored_ceiling_bytes", lambda *a, **k: 1_000_000.0)
    monkeypatch.setattr(_pm, "read_stage_ms", lambda *a, **k: measured)
    monkeypatch.setattr(_pm, "_read_baseline_profile", lambda: {})
    monkeypatch.setattr(_pm, "_summary_mod", lambda: _Roofs(roofs))


def test_a_stack_above_its_band_is_reported(monkeypatch):
    _arrange(
        monkeypatch,
        {"prompt": {"binds": "memory", "memory_ms": 20.91}},
        {"prompt": 182.44},
    )
    out = _pm._stages_short_of_achievable()
    assert [r["stage"] for r in out] == ["prompt"]
    # band low fraction 0.60 -> the slowest time still inside the band is roof / 0.60
    assert abs(out[0]["achievable_ms"] - 20.91 / 0.60) < 0.01
    assert out[0]["over_by_ms"] > 147


def test_a_stack_that_beat_its_band_is_not_reported(monkeypatch):
    """Going past the band is the goal, not a fault."""
    _arrange(monkeypatch, {"recurring": {"binds": "memory", "memory_ms": 16.92}}, {"recurring": 10.99})
    assert _pm._stages_short_of_achievable() == []


def test_the_worst_offender_comes_first(monkeypatch):
    _arrange(
        monkeypatch,
        {
            "a": {"binds": "memory", "memory_ms": 20.0},
            "b": {"binds": "compute", "compute_ms": 6.0},
        },
        {"a": 182.0, "b": 38.0},
    )
    assert [r["stage"] for r in _pm._stages_short_of_achievable()] == ["a", "b"]


def test_the_roof_follows_what_the_stage_binds_on(monkeypatch):
    """A compute-bound stage is judged against its compute roof, not its memory one."""
    _arrange(monkeypatch, {"s": {"binds": "compute", "memory_ms": 2.77, "compute_ms": 6.49}}, {"s": 9.0})
    out = _pm._stages_short_of_achievable()
    assert not out, "9.0 is inside 6.49/0.60 = 10.82; judging it against the memory roof would fail it"


def test_the_band_is_the_models_own_not_a_constant(monkeypatch):
    """An MoE bands at 37.5-50%; the dense 60-80 pair would hold it to a bar it cannot reach."""
    _arrange(monkeypatch, {"s": {"binds": "memory", "memory_ms": 10.0}}, {"s": 25.0}, band=(37.5, 50.0))
    assert _pm._stages_short_of_achievable() == [], "10.0/0.375 = 26.7, so 25.0 is inside this model's band"


def test_nothing_priceable_blocks_nothing(monkeypatch):
    for roofs, measured in (({}, {"s": 9.0}), ({"s": {"binds": "memory"}}, {"s": 9.0}), ({"s": {}}, {})):
        _arrange(monkeypatch, roofs, measured)
        assert _pm._stages_short_of_achievable() == [], (roofs, measured)


def test_no_recoverable_unit_means_no_bar(monkeypatch):
    """The registry's rule: never default the unit. A bar nobody can compute must not block a run."""
    _arrange(monkeypatch, {"s": {"binds": "memory", "memory_ms": 10.0}}, {"s": 900.0})
    monkeypatch.setattr(_pm, "_load_perf_target_inputs", lambda *a, **k: None)
    monkeypatch.setattr(_pm, "_anchored_ceiling_facts", lambda *a, **k: None)
    assert _pm._stages_short_of_achievable() == []


def test_the_unit_is_taken_from_the_anchor_when_the_facts_file_omits_it(monkeypatch):
    """THE TRAP. perf_target_inputs.json carries the bytes and the block map and NO unit -- checked
    against a live run, where it returns weight_bytes/total_params/stage_roots and nothing else. A
    `file or anchor` resolves to the file, loses the unit, and the bar silently never fires, which
    looks exactly like a bar that always passes."""
    _arrange(monkeypatch, {"s": {"binds": "memory", "memory_ms": 20.0}}, {"s": 900.0})
    monkeypatch.setattr(_pm, "_load_perf_target_inputs", lambda *a, **k: {"weight_bytes": 1, "total_params": 2})
    monkeypatch.setattr(_pm, "_anchored_ceiling_facts", lambda *a, **k: {"unit": "token"})
    out = _pm._stages_short_of_achievable()
    assert [r["stage"] for r in out] == ["s"], "the unit must come from the anchor the file does not carry"


def test_the_dense_default_is_derived_from_the_band_owner(monkeypatch):
    """0.60 is the product of perf_target's two band constants, not an independent number. A typed
    copy keeps the old physics the day that shape changes -- the failure rate_and_band records, where
    a second hardcoded (0.60, 0.80) had the report and the gate judging one run against 84.0 and 51.2."""
    from agent import perf_target

    src = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("def _stages_short_of_achievable()")
    seg = src[i : i + 2600]
    # CODE only -- a comment may quote the number it is explaining away.
    body = [l for l in seg.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not [l for l in body if "0.60" in l], "the dense band fraction is typed instead of derived"
    assert "_DENSE_BAND_HI" in seg and "_BAND_LO_OF_HI" in seg
    # and the product is the number the band actually uses
    assert abs(perf_target._DENSE_BAND_HI * perf_target._BAND_LO_OF_HI - 0.60) < 1e-9


def test_a_reader_that_raises_does_not_block_the_run(monkeypatch):
    def _boom():
        raise RuntimeError("no throughput")

    monkeypatch.setattr(_pm, "_read_throughput", _boom)
    assert _pm._stages_short_of_achievable() == []


def test_the_gate_vetoes_on_it_and_says_so():
    src = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")
    assert "_short = _stages_short_of_achievable()" in src
    assert "if _short:\n        can_stop = False" in src, "the list must actually veto the stop"
    assert '"stages_short_of_achievable": _short' in src, "a veto the caller cannot see is a silent refusal"
