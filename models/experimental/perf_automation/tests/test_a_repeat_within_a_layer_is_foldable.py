"""Folding repeated evaluation into one wider matmul.

A high call count is NOT evidence: every matmul in a 30-layer model runs 30 times. What separates a
foldable repeat from ordinary per-layer recurrence is running MORE than once per layer -- and the
profile states the baseline itself, because top_ops groups by (op_code, shape, memory) and the modal
count across fingerprints IS once-per-layer."""
import importlib.util
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _mcp():
    spec = importlib.util.spec_from_file_location("_pm_fold", _PA / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _op(code, count, gap=0.0, bucket="matmul"):
    return {"op_code": code, "count": count, "gap_ms": gap, "bucket": bucket, "bound_by": "compute"}


def _prof(ops):
    return {"device_ms": 100.0, "open_ops": ops}


def test_ordinary_per_layer_recurrence_is_not_flagged():
    """THE FALSE POSITIVE THAT WOULD HANG THE RUN. Every matmul running once per layer has a high
    count; if that alone fired, every op would block termination forever."""
    m = _mcp()
    prof = _prof([_op("Matmul QKV", 30, 5.0), _op("Matmul O", 30, 4.0), _op("Matmul FF1", 30, 6.0)])
    assert m._fold_gate(prof, []) is None


def test_a_fingerprint_running_twice_per_layer_is_flagged():
    """A SwiGLU gate/up pair: identical shape, same input, one fingerprint, 2x depth."""
    m = _mcp()
    prof = _prof([_op("Matmul QKV", 30, 5.0), _op("Matmul O", 30, 4.0), _op("Matmul GateUp", 60, 9.0)])
    b = m._fold_gate(prof, [])
    assert b is not None and b["next_rung"] == "structural-fold"
    assert "Matmul GateUp" in b["op"]
    assert "2x per layer" in b["reason"]


def test_a_non_multiple_is_not_flagged():
    """Two stacks of different depth (Voxtral: 32 audio, 30 language) must not read as a fold."""
    m = _mcp()
    prof = _prof([_op("A", 30, 1.0), _op("B", 30, 1.0), _op("C", 30, 1.0), _op("D", 32, 5.0)])
    assert m._fold_gate(prof, []) is None


def test_too_few_fingerprints_to_have_a_mode():
    m = _mcp()
    assert m._fold_gate(_prof([_op("A", 30, 1.0), _op("B", 60, 5.0)]), []) is None


def test_an_immaterial_gap_is_not_worth_blocking_on():
    m = _mcp()
    prof = _prof([_op("A", 30, 1.0), _op("B", 30, 1.0), _op("C", 90, 0.0001)])
    assert m._fold_gate(prof, []) is None


def test_it_clears_only_on_a_measured_win(monkeypatch):
    m = _mcp()
    monkeypatch.setattr(m, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda a: bool(a.get("won")))})())
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    prof = _prof([_op("A", 30, 1.0), _op("B", 30, 1.0), _op("C", 90, 5.0)])
    assert m._fold_gate(prof, [{"kernel_kind": "fold", "won": False}]) is not None
    assert m._fold_gate(prof, [{"kernel_kind": "fold", "won": True}]) is None


def test_not_foldable_is_a_legitimate_outcome(monkeypatch):
    """The gate sees recurrence, not concatenability. 'none: <evidence>' must be able to end it."""
    m = _mcp()
    monkeypatch.setattr(m, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda _a: False)})())
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    prof = _prof([_op("A", 30, 1.0), _op("B", 30, 1.0), _op("C", 90, 5.0)])
    assert m._fold_gate(prof, [{"kernel_kind": "fold"}] * 3) is None
    assert "none: <why not foldable>" in m._fold_gate(prof, [])["reason"]


def test_it_is_wired_into_the_stop_gate():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("conv_block = _conv_gate(prof, attempts)")
    assert "_fold_gate(prof, attempts)" in src[i : i + 400]
