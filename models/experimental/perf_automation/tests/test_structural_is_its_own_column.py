"""The ladder has eight rungs and the report showed six of them.

perf_mcp._RUNG_PRIORITY is the climb order -- grid, fidelity, dtype, shard, host, structural,
tt-lang, cpp -- and `structural` had no column: _HOST_KINDS folded it into `host`, so an algorithmic
restructure and a trace/dispatch fix rendered identically. Run 13 shows ReshapeView and
TilizeWithValPadding with a `host` win while the ledger records both as `structural`. `tp-fracture`
fell to the anonymous `other` bucket, and so did every specific lever the gates mint."""
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from cc_optimize.summary import _LEVEL_COLS, _level_of  # noqa: E402


def test_every_rung_the_ladder_mints_has_a_column():
    """The two lists must not disagree -- that disagreement IS the defect."""
    from cc_optimize.perf_mcp import _RUNG_PRIORITY

    for rung in set().union(*(set(v) for v in _RUNG_PRIORITY.values())):
        assert rung in _LEVEL_COLS, "%r is on the ladder but has no report column" % rung
    assert "tp-fracture" in _LEVEL_COLS, "tp-fracture is mintable as a next_rung but has no column"


def test_structural_is_not_swallowed_by_host():
    assert _level_of("structural") == "structural"
    assert _level_of("trace") == "host", "host lost the dispatch levers it actually owns"


def test_each_gate_lever_lands_in_structural_not_other():
    """Four gates mint four distinct kinds; all of them resolved to `other`, so a conv weight prep
    was indistinguishable from a KV-cache in the report."""
    for kind in (
        "structural-conv",
        "structural-fold",
        "structural-order",
        "structural-decode",
        "kv-cache",
        "conv-prep",
    ):
        assert _level_of(kind) == "structural", "%s still falls into other" % kind


def test_an_unknown_name_still_goes_to_other():
    """`other` must keep meaning 'unclassifiable', not become a dumping ground with no members."""
    assert _level_of("something-nobody-has-written-yet") == "other"


def test_the_legend_describes_the_new_columns():
    from cc_optimize.summary import _LEVEL_SEMANTICS

    assert "structural =" in _LEVEL_SEMANTICS
    assert "tp-fracture =" in _LEVEL_SEMANTICS
