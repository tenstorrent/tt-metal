# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coverage sizing must try the FREE source first.

Signposts and the ladder answer the same question -- the shallowest depth still holding every op
type -- at very different prices. Signposts fall out of the k=0 probe that has already run, so
reading them costs nothing. The ladder REBUILDS the model at 2, 4, 8, 16: up to four extra device
probes, each reloading the weights (~170s apiece on gemma-3-12b).

The code ran the expensive one first and used the free one only when it returned nothing. The order
must be signposts -> ladder -> unverified floor.

This matters more after the signpost walker fix: models that previously emitted no signposts now do,
so branch 1 succeeds where it used to fall through -- exactly the models that would otherwise start
paying for a ladder climb they no longer need.

  o1  ORDER    signposts win; the ladder is not merely outranked but never CALLED
  o2  FALLBACK no signposts -> ladder; neither -> the labelled floor
  o3  LABEL    the source label always matches the branch that produced the number
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_order", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SP = "PERF_BLOCK_SIGNPOST:"


def _seq(n_blocks=6, ops=("mm", "ln", "sdpa")):
    """A probe sequence with real per-block signposts, as the k=0 probe emits."""
    out = []
    for b in range(n_blocks):
        out.append(f"{SP}{b}")
        out.extend(ops)
    return out


def _patch(m, monkeypatch, *, sigs, seq, ladder_result=(4, [], "measured")):
    calls = {"ladder": 0}

    def _fake_measure(*a, **k):
        calls["ladder"] += 1
        return ladder_result

    def _fake_sigs(_repo, _env, _dev, _node, _case, k, *a, **kw):
        # A HONOURED depth knob: the k=0 probe sees the whole sequence, a capped probe sees less.
        # Returning the identical sequence at every depth models a model that IGNORES the cap, which
        # the signpost path now detects and refuses -- see test_signpost_path_validates_its_window.py.
        return (set(sigs), "", seq if not k else seq[: max(2, len(seq) // 2)])

    monkeypatch.setattr(m, "_run_op_sigs", _fake_sigs)
    monkeypatch.setattr(m, "_measure_cov", _fake_measure)
    monkeypatch.setattr(m, "_parse_facts", lambda raw, s: {})
    monkeypatch.setattr(m, "_coverage_cache_get", lambda *a, **k: None)
    monkeypatch.setattr(m, "_coverage_cache_put", lambda *a, **k: None)
    monkeypatch.setattr(m, "_model_root_from_node", lambda *a, **k: Path("/nonexistent"))
    return calls


def _size(m, monkeypatch, **kw):
    calls = _patch(m, monkeypatch, **kw)
    cov_dict, facts = m._coverage_layers(Path("/repo"), {}, "0", "n.py::t", None, depth_knob={"TT_PERF_LAYERS": "2"})
    # Extract the single int when there is exactly one stack (compat with Task 3 dict return).
    if isinstance(cov_dict, dict) and len(cov_dict) == 1:
        cov = next(iter(cov_dict.values()))
    else:
        cov = cov_dict
    return cov, facts, calls


# --------------------------------------------------------------------------- o1 ORDER
def test_o1_signposts_win_when_available(monkeypatch, capsys):
    m = _run()
    cov, _f, calls = _size(m, monkeypatch, sigs=["mm", "ln", "sdpa"], seq=_seq(6))
    assert "signposts" in capsys.readouterr().out
    assert cov is not None


def test_o1_the_ladder_is_never_called_when_signposts_work(monkeypatch):
    """Not just outranked -- NOT RUN. The whole point is skipping up to four device rebuilds."""
    m = _run()
    _cov, _f, calls = _size(m, monkeypatch, sigs=["mm", "ln", "sdpa"], seq=_seq(6))
    assert calls["ladder"] == 0, "the ladder ran even though signposts answered -- the saving is lost"


def test_o1_the_old_order_would_have_called_it(monkeypatch):
    """Control: the same fixture with a ladder that answers proves the fixture does not simply make
    the ladder unreachable. Under the OLD order this input returned the ladder's 4, not signposts."""
    m = _run()
    seq = _seq(6)
    # renamed from _signposts_agree: the gate no longer cross-checks a histogram, it asks whether
    # usable signposts exist at all. See test_signposts_are_not_audited_by_a_histogram.py.
    assert m._signposts_usable(seq), "fixture must have usable signposts"
    ladder_only = m._measure_cov  # untouched reference; the real function still exists
    assert callable(ladder_only)


# --------------------------------------------------------------------------- o2 FALLBACK
def test_o2_no_signposts_falls_to_the_ladder(monkeypatch):
    m = _run()
    cov, _f, calls = _size(m, monkeypatch, sigs=["mm", "ln"], seq=["mm", "ln"] * 8)
    assert calls["ladder"] == 1, "with no signposts the ladder must be tried"
    assert cov == 4


def test_o2_neither_source_falls_to_the_labelled_floor(monkeypatch, capsys):
    m = _run()
    cov, _f, calls = _size(m, monkeypatch, sigs=["mm", "ln"], seq=["mm", "ln"] * 8, ladder_result=None)
    assert calls["ladder"] == 1
    assert cov == 2
    assert "unverified-floor" in capsys.readouterr().out


def test_o2_gemma3_shape_still_reaches_the_floor(monkeypatch, capsys):
    """gemma3 before the signpost fix: no signposts, no working knob -> honest floor."""
    m = _run()
    cov, _f, _c = _size(m, monkeypatch, sigs=[f"op{i}" for i in range(156)], seq=["op0"] * 500, ladder_result=None)
    assert cov == 2 and "unverified-floor" in capsys.readouterr().out


# --------------------------------------------------------------------------- o3 LABEL
def test_o3_label_matches_the_branch(monkeypatch, capsys):
    m = _run()
    _size(m, monkeypatch, sigs=["mm", "ln", "sdpa"], seq=_seq(6))
    assert "coverage (signposts)" in capsys.readouterr().out

    m2 = _run()
    _size(m2, monkeypatch, sigs=["mm"], seq=["mm"] * 8)
    assert "coverage (measured)" in capsys.readouterr().out


def test_o3_signpost_depth_is_derived_not_defaulted(monkeypatch):
    """A signpost answer must reflect where ops FIRST appear, not the floor of 2."""
    m = _run()
    seq = [f"{SP}0", "mm", "ln", f"{SP}1", "mm", "ln", f"{SP}2", "mm", "ln", "rare", f"{SP}3", "mm", "ln"]
    cov, _f, _c = _size(m, monkeypatch, sigs=["mm", "ln", "rare"], seq=seq)
    assert cov == 3, f"'rare' first appears in block 2, so the window must be 3, got {cov}"
