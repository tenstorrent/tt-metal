"""Holding every core says nothing about how the work is carved across them.

The grid rung's whole instruction was "full-grid program_config." -- one step, no ladder, unlike
dtype (bf16 -> bf8_b -> bf4_b) or fidelity (HiFi4 -> HiFi2 -> LoFi). So an op that already held the
grid had the rung satisfied and the ladder moved on.

On voxtral_mini_3b_2507 (2026-09-04) that is where prefill and encode are stuck. The prefill
projection runs on all 110 cores at the lowest fidelity and is still ~12x off its floor, because
naming a core_grid leaves ttnn a 1-D multicast with 1x1 subblocks. Forty-eight prefill attempts
across every existing rung produced three wins. The run did find it -- one win was an out_subblock
table hand-written into the model -- but a hand edit is not a rung, so the next op starts from
nothing.

Decode is exempt by shape, not by name: its projections are one tile row tall, which can be carved
exactly one way, and they already sit at ~1.1x their floor.
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
    spec = importlib.util.spec_from_file_location("pmcp_block_rung", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _code_of(fn: str) -> str:
    """A function's CODE, with its docstring removed.

    These checks kept reading prose as if it were code: the explanation of why a blanket except is
    wrong contains the words "except Exception", and a longer docstring pushed the import out of a
    fixed-size window. What is being asserted is what the function DOES.
    """
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("def %s" % fn)
    body = src[i : src.index("\ndef ", i + 10)]
    q = body.find('"""')
    return body[body.index('"""', q + 3) + 3 :] if q != -1 else body


def _op(code, grid="full", bound="memory"):
    return {
        "op_code": code,
        "shape": code,
        "grid": grid,
        "bound_by": bound,
        "weight_dtype": "bf8_b",
        "fidelity": "lofi",
    }


def test_a_full_grid_matmul_is_asked_to_shape_the_work():
    """The state the old ladder had no answer for: every core held, still far off the floor."""
    m = _pm()
    code = "MatmulDeviceOperation 512 x 3072 x 8192"
    assert m._op_ladder_status(_op(code), code, [])[1] == "knob:block"


def test_a_matmul_one_tile_row_tall_is_never_sent_there():
    """One tile row can be carved one way. Decided from the op's shape, not from a stage name."""
    m = _pm()
    code = "MatmulDeviceOperation 32 x 3072 x 8192"
    assert m._op_ladder_status(_op(code), code, [])[1] != "knob:block"


def test_the_grid_still_comes_first_when_it_is_not_full():
    """Shaping work across cores you have not claimed yet is the wrong order."""
    m = _pm()
    code = "MatmulDeviceOperation 512 x 3072 x 8192"
    assert m._op_ladder_status(_op(code, grid="partial"), code, [])[1] == "knob:grid"


def test_an_op_that_is_not_a_matmul_is_not_sent_there():
    m = _pm()
    code = "LayerNormDeviceOperation"
    assert m._op_ladder_status(_op(code), code, [])[1] != "knob:block"


def test_the_rung_sits_between_occupying_and_the_precision_knobs():
    """It completes the grid decision, so it is offered before precision is spent."""
    m = _pm()
    for bound in ("memory", "compute", "dispatch", ""):
        order = m.ladder_order(bound)
        assert order.index("grid") < order.index("block") < order.index("fidelity"), bound
        assert order.index("block") < order.index("dtype"), bound


def test_m_is_read_from_the_shape_the_op_reports():
    m = _pm()
    assert m._matmul_m_tiles({"op_code": "MatmulDeviceOperation 32 x 3072 x 8192"}) == 1
    assert m._matmul_m_tiles({"op_code": "MatmulDeviceOperation 512 x 3072 x 8192"}) == 16
    assert m._matmul_m_tiles({"op_code": "MatmulDeviceOperation 1504 x 1280 x 5120"}) == 47
    assert m._matmul_m_tiles({"op_code": "LayerNormDeviceOperation"}) == 0
    assert m._matmul_m_tiles(None) == 0


def test_the_tile_height_is_not_redefined_here():
    """agent.tp owns TILE. A second copy is a second thing to get wrong."""
    code = _code_of("_matmul_m_tiles")
    assert "from agent.tp import TILE" in code
    assert "= 32" not in code, "a local tile size was introduced"


def test_every_knob_is_counted_the_same_way():
    """The four hand-written per-knob counters were one copy each; a fifth would never saturate."""
    m = _pm()
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    assert "grid_tries = sum(" not in src, "a hand-written per-knob counter came back"
    code = "MatmulDeviceOperation 512 x 3072 x 8192"
    tried = [{"kernel_kind": "knob:block", "op_signature": code} for _ in range(9)]
    assert m._op_ladder_status(_op(code), code, tried)[1] != "knob:block", "the rung never saturates"


def test_the_knob_set_has_one_definition():
    m = _pm()
    assert set(m._KNOB_RUNG_NAMES) == set(m._KNOBS)
    assert "block" in m._KNOBS


def test_the_report_has_a_column_for_it():
    """An attempt on a rung the report cannot render is an attempt nobody can see."""
    from cc_optimize import summary

    assert "block" in summary._LEVEL_COLS
    assert summary._level_of("knob:block") == "block"
    assert summary._level_of("block") == "block"


def test_the_agent_is_told_what_the_rung_means():
    """A rung the ladder offers and the prompt never explains is a rung the agent improvises."""
    src = (_CC / "run.py").read_text(encoding="utf-8")
    assert "knob:block" in src
    i = src.index("knob:block")
    seg = src[i : i + 900]
    for term in ("in0_block_w", "out_subblock", "per_core_M", "transpose_mcast"):
        assert term in seg, term


def test_no_stage_name_is_typed_into_the_rung():
    """Which ops qualify is decided by shape; a stage name here would outlive the model that had it."""
    code = _code_of("_matmul_m_tiles")
    for typed in ("decode", "prefill", "encode"):
        assert '"%s"' % typed not in code, typed


def test_the_shape_reader_does_not_swallow_its_own_bugs():
    """0 withdraws the rung, so a blanket except turns a typo into "nothing to carve", silently.

    That is not hypothetical: `re` is imported as `_re` in this module, the bare name raised
    NameError, and `except Exception: return 0` answered 0 for every shape while the suite stayed
    green -- because the cases asserted the rung was offered, not the count.
    """
    code = _code_of("_matmul_m_tiles")
    assert "except Exception" not in code, "a blanket except is back; a typo here disables the rung"
    assert "except ImportError" in code, "the one thing that legitimately varies must still be caught"


def test_a_shape_that_cannot_be_read_is_zero_not_a_crash():
    """The real runtime variations still degrade quietly."""
    m = _pm()
    assert m._matmul_m_tiles({}) == 0
    assert m._matmul_m_tiles(None) == 0
    assert m._matmul_m_tiles("not a dict") == 0
    assert m._matmul_m_tiles({"op_code": None}) == 0
