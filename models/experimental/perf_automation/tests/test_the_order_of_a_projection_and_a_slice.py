"""Adjacency was in the capture and was being discarded.

Report rows arrive in execution order (GLOBAL CALL COUNT, which _raw_index already joins on) and
grouping by op_class threw that away -- so nothing could ask "what does this op feed into", even
though the playbook teaches exactly that technique by hand (09 section 5)."""
import importlib.util
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _mcp():
    spec = importlib.util.spec_from_file_location("_pm_order", _PA / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the neighbour pass


def test_neighbours_are_the_most_common_pairing_not_a_one_off():
    """An op appearing once per layer has the same neighbours every layer, so the mode is the
    structural answer; an odd pairing at a stack boundary is noise."""
    import sys

    sys.path.insert(0, str(_PA))
    from agent.tracy_tool import _neighbours

    a, b, c = ("Matmul", "s", "m"), ("Slice", "s", "m"), ("Norm", "s", "m")
    ordered = [(c, "Norm"), (a, "Matmul"), (b, "Slice")] * 4 + [(a, "Matmul"), (c, "Norm")]
    nb = _neighbours(ordered)
    assert nb[a]["next"] == "Slice", "the dominant successor was not found"
    assert nb[a]["prev"] == "Norm"


def test_order_is_preserved_from_the_capture_into_op_rows():
    """The join point: build_buckets computes neighbours before grouping, _top_ops carries them."""
    src = (_PA / "agent" / "tracy_tool.py").read_text()
    assert "_nbrs = _neighbours(_ordered)" in src
    assert '"prev_op": _nb.get("prev", "")' in src
    roof = (_PA / "agent" / "roofline.py").read_text()
    assert '"prev_op": o.get("prev_op")' in roof, "roofline drops them before open_ops"


# ---------------------------------------------------------------- the gate


def _op(code, nxt="", prv="", gap=5.0):
    return {"op_code": code, "next_op": nxt, "prev_op": prv, "gap_ms": gap, "bucket": "matmul", "count": 30}


def _prof(ops):
    return {"device_ms": 100.0, "open_ops": ops}


def test_a_projection_feeding_a_slice_is_flagged():
    m = _mcp()
    b = m._order_gate(_prof([_op("Matmul QKV", nxt="SliceDeviceOperation")]), [])
    assert b is not None and b["next_rung"] == "structural-order"
    assert "Matmul QKV" in b["op"]


def test_a_projection_with_no_slice_neighbour_is_not_flagged():
    m = _mcp()
    assert m._order_gate(_prof([_op("Matmul QKV", nxt="LayerNorm", prv="LayerNorm")]), []) is None


def test_a_non_projection_next_to_a_slice_is_not_flagged():
    """The lever is about projection-vs-slice ordering, not any op that happens to precede a slice."""
    m = _mcp()
    assert m._order_gate(_prof([{**_op("LayerNorm", nxt="Slice"), "op_code": "LayerNorm"}]), []) is None


def test_a_forced_order_is_a_legitimate_outcome(monkeypatch):
    m = _mcp()
    monkeypatch.setattr(m, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda _a: False)})())
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    prof = _prof([_op("Matmul QKV", nxt="Slice")])
    assert "order is forced" in m._order_gate(prof, [])["reason"]
    assert m._order_gate(prof, [{"kernel_kind": "order"}] * 3) is None


def test_it_clears_only_on_a_measured_win(monkeypatch):
    m = _mcp()
    monkeypatch.setattr(m, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda a: bool(a.get("won")))})())
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    prof = _prof([_op("Matmul QKV", nxt="Slice")])
    assert m._order_gate(prof, [{"kernel_kind": "order", "won": False}]) is not None
    assert m._order_gate(prof, [{"kernel_kind": "order", "won": True}]) is None


def test_it_is_wired_into_the_stop_gate():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("fold_block = _fold_gate(prof, attempts)")
    assert "_order_gate(prof, attempts)" in src[i : i + 400]
