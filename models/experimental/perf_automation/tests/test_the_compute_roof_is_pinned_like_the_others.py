# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The compute roof moved when the model got faster, because its denominator was never anchored.

THREE INPUTS BUILD THE ROOFLINE. Two were pinned and one was not:

    memory   active_bytes ─→ KIND_ACTIVE_BYTES @ PHASE_BEFORE ─→ write-once      PINNED
    floor    modeled_floor_ms ─→ KIND_FLOOR, pinned where produced               PINNED
    compute  peak_flops ─→ dominant fidelity of the CURRENT picture              floated

_promote_baseline states the principle itself, and then breaks it in the same function:

    out = dict(prof)                 # the NEW profile -- ops, shapes, FIDELITY
    if kept_ms is not None:
        out["device_ms"] = kept_ms   # only the ms is ratcheted
    "[baseline-ratchet] picture refreshed, BAR unchanged
     (a re-profile must not redefine what wins are graded against)"

The bar is protected; the picture the ceiling is read from is replaced. Blackhole's modes are 4x
apart (LoFi 702, HiFi2 351, HiFi3 234, HiFi4 175.5 TFLOPS), so the `fidelity` rung moves the roof
every time it lands. Measured on voxtral's run-18 profile: hifi4 carries 4.037e12 FLOPs against
hifi2's 3.299e12 -- a 5.0%-of-total margin, and the largest single hifi4 matmul is 16.7%. One op
changing mode doubles the ceiling.

WHY IT OUTLIVED THE OTHER TWO. Nothing impossible ever printed. The memory roof divides by a fixed
512 GB/s, so drift there shows up as a bandwidth above peak and gets caught -- that is how the byte
anchor was found. Here the peak IS what moves, and the measurement moves with it:

    before   ceiling 66.78 ms   measured 106.24 ms   62.9%
    after    ceiling 33.39 ms   measured  53.10 ms   62.9%   <- 2x faster, identical reading

The error and the win cancel exactly, so the reader sees neither.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """A PER-MODEL keyed ledger in a scratch dir, plus the two modules under test.

    conftest points PERF_MCP_LEDGER at ONE file, which is right for most tests and collapses the
    (model, task) keying these need -- two models then share an anchor and write-once makes the
    second read the first's value. Dropping the var and keying ledger_path is what
    test_single_source_of_truth does for the same reason.
    """
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    import cc_optimize.summary as S

    # SUMMARY'S OWN LEDGER INSTANCE. _ledger() loads measurements.py BY PATH under the name
    # "tt_measurements", so it is a different module object from `import cc_optimize.measurements`
    # -- patching the latter leaves the reader looking at an unpatched path, and the anchor the test
    # wrote is invisible to the code under test.
    led = S._ledger()
    monkeypatch.setattr(
        led, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )
    return S, led


# NOT tests/data/ -- the repo's .gitignore has a bare `data/` rule, so a fixture parked there is
# silently never committed and the tests fail on a fresh clone while passing for whoever wrote them.
_FIXTURE = _PA / "tests" / "profiles" / "run18_baseline_profile.json"


def _profile(fidelity):
    """Voxtral run 18's REAL baseline profile, relabelled to the named mode.

    Measured data rather than a synthetic op: residual_report prices an op from a set of annotator
    fields that is easy to get subtly wrong by hand, and an unpriced op yields an empty breakdown --
    which looks exactly like "the ceiling did not move".

    Relabelling is the whole experiment. Same shapes, same FLOPs, same bytes; only the math mode
    changes, which is precisely what the `fidelity` rung does.
    """
    p = json.loads(_FIXTURE.read_text())
    for b in p.get("buckets", []):
        for o in b.get("top_ops") or []:
            if str(o.get("fidelity", "")).startswith("hifi"):
                o["fidelity"] = fidelity
    return p


def _peaks(S, prof, model):
    r = S._stage_roofs(4_777_269_892, 512.0, 1, "token", prof, {"decode": 17.25}, model=model, task="main")
    d = r.get("decode") or {}
    return (d.get("peak_flops") or 0) / 1e12, (d.get("peak_flops_now") or 0) / 1e12


def test_the_roof_no_longer_moves_when_the_math_mode_does(wired):
    """THE WHOLE POINT. Same model, same FLOPs, one relabel -- the ceiling must not move."""
    S, M = wired
    from agent.environment import ARCH_FACTS
    from agent.perf_target import chip_peak_flops as cpf

    M.anchor(M.KIND_PEAK_FLOPS, cpf(ARCH_FACTS["blackhole"], "hifi4"), depth="token", source="t", model="m")
    hi, _ = _peaks(S, _profile("hifi4"), "m")
    lo, _ = _peaks(S, _profile("hifi2"), "m")
    assert abs(hi - lo) < 1e-6, "the roof still floats: %.1f vs %.1f TFLOPS" % (hi, lo)
    assert abs(hi - 175.5) < 0.1, hi


def test_without_the_pin_it_really_did_move(wired):
    """The defect itself, kept as a test so the fix cannot be quietly reverted."""
    S, _M = wired
    hi, _ = _peaks(S, _profile("hifi4"), "never-pinned")
    lo, _ = _peaks(S, _profile("hifi2"), "never-pinned")
    assert abs(hi - 175.5) < 0.1 and abs(lo - 351.0) < 0.1, (hi, lo)


def test_the_fidelity_win_is_still_visible(wired):
    """Pinning must not HIDE the win. A ceiling that never moves and a build that got faster are two
    facts, and the report needs both -- otherwise pinning trades an invisible error for an invisible
    improvement."""
    S, M = wired
    from agent.environment import ARCH_FACTS
    from agent.perf_target import chip_peak_flops as cpf

    M.anchor(M.KIND_PEAK_FLOPS, cpf(ARCH_FACTS["blackhole"], "hifi4"), depth="token", source="t", model="m")
    _, now_hi = _peaks(S, _profile("hifi4"), "m")
    _, now_lo = _peaks(S, _profile("hifi2"), "m")
    assert abs(now_hi - 175.5) < 0.1 and abs(now_lo - 351.0) < 0.1, (now_hi, now_lo)


def test_nothing_pinned_still_renders_a_ceiling(wired):
    """Every existing path stays alive: a report drawn before any anchor exists still gets a roof."""
    S, _M = wired
    pk, now = _peaks(S, _profile("hifi4"), "fresh")
    assert pk > 0 and abs(pk - now) < 1e-6


def test_the_anchor_is_write_once(wired):
    """Second write must not move it -- the property the other two anchors already have."""
    S, M = wired
    M.anchor(M.KIND_PEAK_FLOPS, 175.5e12, depth="token", source="t", model="w")
    M.anchor(M.KIND_PEAK_FLOPS, 702.0e12, depth="token", source="t", model="w")
    assert abs(M.anchor_value(M.KIND_PEAK_FLOPS, depth="token", model="w", task="main") - 175.5e12) < 1


def test_it_is_keyed_on_the_unit_not_the_profiling_window(wired):
    """A peak is a property of the math mode and the silicon. Keying it on the layer window -- which
    is what the FLOOR is keyed on, because a floor sums profiled ops -- would pin a different roof
    for every coverage depth."""
    S, M = wired
    M.anchor(M.KIND_PEAK_FLOPS, 175.5e12, depth="token", source="t", model="u")
    assert M.anchor_value(M.KIND_PEAK_FLOPS, depth="token", model="u", task="main") is not None
    assert M.anchor_value(M.KIND_PEAK_FLOPS, depth="2", model="u", task="main") is None


def test_anchors_are_per_model(wired):
    S, M = wired
    M.anchor(M.KIND_PEAK_FLOPS, 175.5e12, depth="token", source="t", model="a")
    M.anchor(M.KIND_PEAK_FLOPS, 702.0e12, depth="token", source="t", model="b")
    assert abs(M.anchor_value(M.KIND_PEAK_FLOPS, depth="token", model="a", task="main") - 175.5e12) < 1
    assert abs(M.anchor_value(M.KIND_PEAK_FLOPS, depth="token", model="b", task="main") - 702.0e12) < 1


def test_the_producer_derives_the_dominant_mode_by_flop_share(tmp_path, monkeypatch):
    """Not the op COUNT and not the first one seen: the mode carrying the most FLOPs."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    import cc_optimize.perf_mcp as pm

    rep = {"open_ops": [{"fidelity": "hifi4", "flops": 1}, {"fidelity": "lofi", "flops": 1_000_000}]}
    assert abs(pm._dominant_peak_flops(rep) / 1e12 - 702.0) < 0.1
    rep2 = {"open_ops": [{"fidelity": "hifi4", "flops": 1_000_000}, {"fidelity": "lofi", "flops": 1}]}
    assert abs(pm._dominant_peak_flops(rep2) / 1e12 - 175.5) < 0.1


def test_a_report_with_no_flops_pins_nothing(tmp_path, monkeypatch):
    """0.0 rather than a guess, so a run with nothing to measure leaves the renderer as it was."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    import cc_optimize.perf_mcp as pm

    assert pm._dominant_peak_flops({}) == 0.0
    assert pm._dominant_peak_flops({"open_ops": []}) == 0.0
    assert pm._dominant_peak_flops({"open_ops": [{"fidelity": "lofi"}]}) == 0.0


def test_the_real_run18_profile_flips_on_one_op(wired):
    """The margin is not theoretical. Run 18's own baseline: hifi4 4.037e12 vs hifi2 3.299e12, so
    0.369e12 -- 5.0% of the profiled FLOPs -- decides a 2x ceiling, and one matmul carries 16.7%."""
    S, _M = wired
    base = json.loads(_FIXTURE.read_text())
    flipped = copy.deepcopy(base)
    for b in flipped.get("buckets", []):
        for o in b.get("top_ops") or []:
            if str(o.get("fidelity")) == "hifi4":
                o["fidelity"] = "hifi2"
                break
        break
    a, _ = _peaks(S, base, "r18")
    b2, _ = _peaks(S, flipped, "r18")
    assert a != b2, "one op no longer flips the unpinned roof -- has the margin changed?"
