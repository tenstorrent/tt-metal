"""The conv lever, and the three ways a gate like it goes wrong.

`conv_pool` is a first-class op_class -- opclass.py maps Conv/Halo/Pool/GridSample/Upsample to it and
STRUCTURAL_OP_CLASSES contains it -- but the playbook tags it in exactly two sections, both of them
the kernel rungs. So a conv op's only routable guidance was "author a custom kernel", for work that
ttnn.prepare_conv_weights does once at load. Voxtral-Mini-3B-2507 has audio_tower.conv1/conv2."""
import importlib.util
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _mcp():
    spec = importlib.util.spec_from_file_location("_pm_conv", _PA / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prof(convs=True, gap=5.0):
    ops = [{"bucket": "matmul", "op_code": "Matmul", "gap_ms": 3.0}]
    if convs:
        ops.append({"bucket": "conv_pool", "op_code": "Conv2d 80x1280", "gap_ms": gap})
    return {"device_ms": 100.0, "open_ops": ops}


def test_a_model_with_convs_is_blocked_until_they_are_prepared():
    m = _mcp()
    b = m._conv_gate(_prof(), [])
    assert b is not None, "a model with material conv gap was not asked to prepare its weights"
    assert b["next_rung"] == "structural-conv"
    assert "prepare_conv_weights" in b["reason"]


def test_a_model_with_no_convs_is_never_asked():
    """THE FAILURE _decode_gate ALREADY PAID FOR. It blocked an encoder-only model on a KV-cache it
    could not have, and only the attempt cap released it -- three rewrites later. A gate that cannot
    say 'not applicable' is a hang with extra steps."""
    m = _mcp()
    assert m._conv_gate(_prof(convs=False), []) is None


def test_an_immaterial_conv_gap_is_not_worth_blocking_on():
    m = _mcp()
    assert m._conv_gate(_prof(gap=0.0001), []) is None


def test_a_kernel_attempt_does_not_clear_it():
    """A hand-written kernel still pays per-call weight preparation, so it cannot resolve this."""
    m = _mcp()
    for kind in ("tt-lang", "cpp", "structural"):
        assert m._conv_gate(_prof(), [{"kernel_kind": kind, "beat_baseline": True}]) is not None, kind


def test_it_clears_only_on_a_measured_win(monkeypatch):
    """The host rung cleared on 'attempted' and one unhelpful try sealed the axis for 158 rounds."""
    m = _mcp()
    monkeypatch.setattr(m, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda a: bool(a.get("won")))})())
    assert m._conv_gate(_prof(), [{"kernel_kind": "conv-prep", "won": False}]) is not None, "cleared on attempted"
    assert m._conv_gate(_prof(), [{"kernel_kind": "conv-prep", "won": True}]) is None, "a measured win did not clear it"


def test_it_yields_after_the_attempt_cap(monkeypatch):
    """An unpreparable conv must not loop forever."""
    m = _mcp()
    monkeypatch.setattr(m, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda _a: False)})())
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    tries = [{"kernel_kind": "conv-prep"}] * 3
    assert m._conv_gate(_prof(), tries) is None, "the cap never releases -- this is a loop"


def test_it_is_wired_into_the_stop_gate():
    """A gate nothing calls is prose. _decode_gate and _host_gate are appended in termination_check;
    this must be too, or it can never block a stop."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("decode_block = _decode_gate(prof, attempts)")
    assert "_conv_gate(prof, attempts)" in src[i : i + 400], "the conv gate is never consulted"
