# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pinned ceiling is the baseline; the floor this build could reach is a second number.

WHY BOTH. Every input to the THEORETICAL column is pinned on purpose: a ceiling recomputed each
round retreats by exactly the factor the measurement improves, and the run is then chasing a target
it can never close. That is correct, and it is not the whole job -- after a few dtype and fidelity
wins the pinned pair describes a model that no longer exists, while still being read as this one's
floor.

Measured on voxtral_mini_3b_2507, 2026-09-10, after four months of banked wins:

    decode    pinned 16.92 ms floor under a  9.99 ms measurement -- 867 GB/s on a 512 GB/s part
    encode    pinned 12.99 ms floor, "in band" at 15.47 ms, while its LoFi floor is 3.25 ms
    prefill   pinned 2.0 B/param and HiFi4 while the build serves bf8_b, bf4_b and LoFi

Two of those three pairs are arithmetically impossible, and nothing in the report said so: a wrong
ceiling renders as a perfectly plausible table. Each of the tool's previous roofline defects had the
same shape and each was found by hand, months apart.

So: the pinned pair stays exactly as it is, a second pair states the floor at the precision actually
being served, and an impossible pair is called impossible where it is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
for _p in (_PA, _PA / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import summary as S  # noqa: E402

_MF = {
    "weight_bytes": 7_222_966_272,
    "total_params": 3_611_483_136,
    "dominant_dtype": "bfloat16",
    "layers": 30,
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "kv_heads": 8,
    "head_dim": 128,
}


def _arrange(monkeypatch, *, bytes_pinned=None, bytes_now=None, peak_pinned=0.0, peak_now=0.0):
    """A build that has moved off its anchors: the pin says one thing, this capture another.

    Patched at the two functions that OWN the distinction -- stage_read_bytes resolves
    pinned -> measured -> estimate, and _peak_for_stage prefers the anchor -- because with no
    ledger present both would otherwise return the same number as their live twin and the
    divergence under test could not arise.
    """
    monkeypatch.setattr(S, "_model_facts", lambda: _MF)
    monkeypatch.setattr(S, "_prompt_tokens", lambda: 416)
    if bytes_pinned is not None:
        monkeypatch.setattr(S, "stage_read_bytes", lambda *a, **k: (bytes_pinned, "pinned"))
    if bytes_now is not None:
        monkeypatch.setattr(S, "_measured_stage_bytes", lambda *a, **k: bytes_now)
    if peak_pinned:
        monkeypatch.setattr(S, "_peak_for_stage", lambda *a, **k: (peak_pinned, ""))
    if peak_now:
        monkeypatch.setattr(S, "_observed_peak_for_stage", lambda *a, **k: (peak_now, "lofi"))


def _roofs(monkeypatch, *, stage_ms=None, **kw):
    _arrange(monkeypatch, **kw)
    return S._stage_roofs(
        active_bytes=int(7.223e9),
        peak_bw_gbps=512.0,
        tp_degree=1,
        unit="tok/s/u",
        profile=None,
        stage_ms=stage_ms,
    )


# ------------------------------------------------------------------ the second pair is computed


def test_the_live_floor_divides_by_this_builds_bytes(monkeypatch):
    """A dtype win shrinks the read set. The pinned roof must not follow it; the live one must."""
    r = _roofs(monkeypatch, bytes_pinned=8_663_871_700, bytes_now={"decode": 4_408_955_872})["decode"]
    assert r["memory_ms"] > r["memory_ms_now"], (r["memory_ms"], r["memory_ms_now"])
    assert abs(r["memory_ms_now"] - (4_408_955_872 / 512e9) * 1000.0) < 0.01


def test_the_live_floor_divides_by_this_builds_peak(monkeypatch):
    """A fidelity win raises the peak. Same rule, on the compute axis."""
    r = _roofs(monkeypatch, peak_pinned=175.5e12, peak_now=702e12)["prefill"]
    assert r["compute_ms"] > r["compute_ms_now"], (r["compute_ms"], r["compute_ms_now"])
    assert abs(r["compute_ms_now"] - (r["flops"] / 702e12) * 1000.0) < 0.01


def test_a_build_still_at_its_baseline_says_nothing_new(monkeypatch):
    """No observed read set and no marked capture: the live pair IS the pinned pair, so the report
    prints exactly what it printed before this existed."""
    for _st, r in _roofs(monkeypatch).items():
        assert r["memory_ms_now"] == r["memory_ms"], _st
        assert r["compute_ms_now"] == r["compute_ms"], _st
        assert r["binds_now"] == r["binds"], _st


def test_the_live_pair_can_bind_differently(monkeypatch):
    """The point of computing it: which resource is the wall is itself a function of precision."""
    r = _roofs(monkeypatch, bytes_pinned=1, peak_pinned=175.5e12, peak_now=702e12)["prefill"]
    assert r["binds"] == "compute", r["binds"]
    assert r["compute_ms_now"] < r["compute_ms"]
    assert r["binds_now"] == "compute"


# ------------------------------------------------------------------------ impossible pairs


def test_a_floor_above_the_measurement_is_recorded_as_impossible(monkeypatch):
    r = _roofs(monkeypatch, stage_ms={"decode": 9.99})["decode"]
    assert r["memory_ms"] > 9.99, "arrange failed: this pair is not the impossible one"
    assert r["physical"] is False


def test_a_floor_below_the_measurement_is_fine(monkeypatch):
    r = _roofs(monkeypatch, stage_ms={"decode": 9999.0})["decode"]
    assert r["physical"] is True


def test_an_unmeasured_stage_states_nothing_either_way(monkeypatch):
    """A pair with one side missing is unchecked, which is not the same as checked and fine."""
    assert _roofs(monkeypatch, stage_ms={})["decode"]["physical"] is None


def test_the_rule_lives_in_one_place():
    from agent.roofline import floor_is_physical

    assert floor_is_physical(16.92, 9.99) is False
    assert floor_is_physical(3.25, 15.47) is True
    assert floor_is_physical(None, 5.0) is None
    assert floor_is_physical(5.0, None) is None


def test_the_provenance_dump_asks_the_same_way():
    """It had the only copy, in a CLI nobody runs on a normal day."""
    src = (_PA / "agent" / "roofline_provenance.py").read_text()
    assert "floor_is_physical" in src, "provenance kept its own copy of the rule"
    assert "bool(roof <= ms)" not in src


# ------------------------------------------------------------------------- and it is printed


def _render(monkeypatch, *, moved=False, **kw):
    if moved:
        # The BINDING roof has to be the one that moved: a peak change says nothing about a floor
        # that memory sets, and the report is right to stay quiet about it.
        _arrange(
            monkeypatch,
            bytes_pinned=8_663_871_700,
            bytes_now={"decode": 4_408_955_872, "prefill": 4_876_431_648},
            peak_pinned=175.5e12,
            peak_now=702e12,
        )
    else:
        _arrange(monkeypatch)
    base = dict(
        unit="tok/s/u",
        theo=42.67,
        band=[25.6, 34.1],
        measured=29.46,
        bw_gbps=354.0,
        peak_bw_gbps=512.0,
        active_bytes=int(7.223e9),
        per_unit_ms=9.99,
        profile=None,
        stage_ms={"decode": 9.99, "prefill": 104.74},
    )
    base.update(kw)
    return "\n".join(S._roofline_tables(**base))


def test_the_report_prints_the_live_floor(monkeypatch):
    text = _render(monkeypatch, moved=True)
    assert "at the precision now served" in text, text[:2000]
    assert "progress reference" in text


def test_a_peak_change_under_a_memory_wall_is_not_announced(monkeypatch):
    """Only the BINDING roof's floor is the stage's floor. A fidelity win on a memory-bound stage
    moves a number that was never the limit, and saying so would be noise."""
    _arrange(monkeypatch, peak_pinned=175.5e12, peak_now=702e12)
    r = S._stage_roofs(active_bytes=int(7.223e9), peak_bw_gbps=512.0, tp_degree=1, unit="tok/s/u", profile=None)[
        "decode"
    ]
    assert r["binds"] == "memory" and r["compute_ms_now"] < r["compute_ms"]
    assert r["memory_ms_now"] == r["memory_ms"]


def test_the_report_calls_an_impossible_pair_impossible(monkeypatch):
    text = _render(monkeypatch)
    assert "IMPOSSIBLE PAIR" in text, text[:2000]
    assert "above the 9.99 ms measured" in text


def test_a_baseline_build_gets_neither_line(monkeypatch):
    """Nothing new to say, so nothing is said -- the table is unchanged for a run that has not
    moved off its anchors."""
    text = _render(monkeypatch, stage_ms={"decode": 9999.0})
    assert "at the precision now served" not in text
    assert "IMPOSSIBLE PAIR" not in text
