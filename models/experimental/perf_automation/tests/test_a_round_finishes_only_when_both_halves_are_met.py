"""A round is finished when every stack is inside its band AND nothing reachable is left.

These were two independent exits and either one ended the round. The one that fired was "every
material op has a recorded attempt", which retired the run's own requirement without meeting it --
so voxtral_mini_3b_2507 (2026-09-05) finished five rounds in a row while prefill sat 101.8 ms above
its band, each ending reported as a clean finish.

And "attempted" meant any attempt at all, on an op with eight rungs. One grid try retired the rest,
and because attempts are cumulative across runs, an op touched weeks earlier stayed retired forever.
Measured on the same model: 48 ops ever attempted, 2.8 of 8 rungs explored on average, 20 of them at
exactly one rung.

The unit is now the rung the gate would hand out next, so each round advances every op by one rung --
the completeness sweep the ladder already describes. It cannot spin, because an op runs out of rungs.

A band that cannot be reached now refuses forever rather than being waived. The round still ends --
the agent exits on its own and no tool prevents that -- but it ends recorded as REFUSED and the run
stops on its budget, which says the work was unfinished instead of claiming it was done.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _pm():
    spec = importlib.util.spec_from_file_location("pmcp_round_both", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_OP = "MatmulDeviceOperation 512 x 3072 x 8192"


def _blocking(rung="knob:block"):
    return [{"op": _OP, "next_rung": rung}]


def _att(kind):
    return [{"op_signature": _OP, "kernel_kind": kind}]


# ---------------------------------------------------------------- the unit is the rung


def test_an_op_owing_a_rung_it_has_not_tried_is_untried():
    """The defect: one grid try retired an op whose block rung had never been touched."""
    m = _pm()
    assert m._untried_material_ops(_blocking("knob:block"), _att("grid")) == [_OP]


def test_an_op_that_has_tried_the_rung_it_is_owed_is_done():
    m = _pm()
    assert m._untried_material_ops(_blocking("knob:block"), _att("knob:block")) == []


def test_the_rung_is_matched_however_it_is_spelled():
    """Rungs are minted prefixed and recorded bare; _normalise_rung already owns that."""
    m = _pm()
    assert m._untried_material_ops(_blocking("knob:block"), _att("block")) == []
    assert m._untried_material_ops(_blocking("block"), _att("knob:block")) == []


def test_last_runs_attempt_does_not_answer_for_this_runs_rung():
    """Attempts are cumulative, so an op touched weeks ago used to stay retired for good."""
    m = _pm()
    old = _att("grid") + _att("dtype")
    assert m._untried_material_ops(_blocking("knob:shard"), old) == [_OP]


def test_an_entry_naming_no_rung_behaves_as_before():
    """A caller that cannot say what is next must not start reporting everything as untried."""
    m = _pm()
    assert m._untried_material_ops([{"op": _OP}], _att("grid")) == []
    assert m._untried_material_ops([{"op": _OP}], []) == [_OP]


def test_an_attempt_on_a_different_shape_does_not_count():
    m = _pm()
    other = [{"op_signature": "MatmulDeviceOperation 32 x 3072 x 8192", "kernel_kind": "knob:block"}]
    assert m._untried_material_ops(_blocking("knob:block"), other) == [_OP]


# ---------------------------------------------------------------- both halves, not either


def test_the_exit_requires_both_halves():
    """`if not _short and not _left` -- either alone used to be enough."""
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("def finish_round")
    body = src[i : src.index("\ndef ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert "if not _short and not _left:" in code
    assert code.count("_record_round_finish") == 2, "a third exit would be a third way to end early"


def test_a_band_alone_no_longer_ends_the_round():
    """The old first exit returned the moment no stack was short, rungs outstanding or not."""
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("def finish_round")
    code = "\n".join(ln for ln in src[i : src.index("\ndef ", i + 10)].splitlines() if not ln.strip().startswith("#"))
    assert "if not _short:\n" not in code, "the band-only exit is back"


def test_the_refusal_says_which_half_is_outstanding():
    """Either half can hold alone now, so always blaming the band would misdescribe the other."""
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("_owed = []")
    seg = src[i : i + 1200]
    assert "if _short:" in seg and "if _left:" in seg
    assert "rung they are next owed" in seg


def test_no_stage_or_rung_name_is_typed_into_the_test():
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("def _untried_material_ops")
    body = src[i : src.index("\ndef ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    code = code[code.index('"""', code.index('"""') + 3) + 3 :]
    for typed in ("decode", "prefill", "encode", "grid", "dtype", "shard", "fidelity"):
        assert '"%s"' % typed not in code, typed
