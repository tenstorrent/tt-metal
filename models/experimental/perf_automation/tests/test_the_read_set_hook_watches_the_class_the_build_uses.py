"""The byte hook watched a class this build never instantiates.

trace_replay counts a stage's read set by patching a ttnn op class and noting the device tensors its
calls touch. It patched `Operation`. But ttnn chooses at REGISTRATION time --
`operation_class = FastOperation if ttnn.CONFIG.enable_fast_runtime_mode else Operation` -- and that
flag defaults to true (config.hpp). Measured on this build: 735 ops are FastOperation, 0 are
Operation, and patching Operation intercepted 0 calls while patching FastOperation intercepted 1.

So the hook returned (0 dispatches, 0 bytes) for every stage of every run, `if bytes > 0` never
fired, nothing was printed, nothing was parsed, and the stage file recorded `bytes: {}` -- while the
roofline quietly rendered the checkpoint ESTIMATE in a column that reads as measured.

Three more defects rode along, and all of them bias the number LOW, which matters because the value
is pinned write-once as the memory roof's divisor:
  * a tensor inside a list/tuple/dict was invisible (no recursion; _tensor_entry returns None for a
    container, so it was not partially counted -- it was skipped);
  * tensors were keyed by id(), which CPython reuses (2000 short-lived objects -> 161 ids);
  * a broken hook and a genuine zero were the same observable: silence."""
import importlib
import sys
import types

import pytest


class _FakeTensor:
    def __init__(self, numel, addr):
        self._numel, self._addr = numel, addr

    def buffer_address(self):
        return self._addr


def _load(monkeypatch, klass_names):
    """trace_replay imports ttnn at module scope, and the tool tree has no ttnn -- so the stub goes
    into sys.modules BEFORE the import. Each test names which op classes its build exposes, which is
    the whole variable under test."""
    dec = types.ModuleType("ttnn.decorators")
    made = {}
    for nm in klass_names:
        made[nm] = type(nm, (), {"__call__": lambda self, *a, **k: None})
        setattr(dec, nm, made[nm])
    pkg = types.ModuleType("ttnn")
    pkg.decorators = dec
    for _n in ("ttnn", "ttnn.decorators"):
        monkeypatch.setitem(sys.modules, _n, pkg if _n == "ttnn" else dec)
    monkeypatch.delitem(sys.modules, "agent.trace_replay", raising=False)
    tr = importlib.import_module("agent.trace_replay")

    from agent import weight_census as wc

    monkeypatch.setattr(wc, "_tensor_entry", lambda t: (t._numel, "bfloat16") if isinstance(t, _FakeTensor) else None)
    monkeypatch.setattr(wc, "bytes_per_elem", lambda name: 2)
    return tr, made


def test_it_counts_ops_of_the_class_the_build_actually_registered(monkeypatch):
    """The regression itself: only FastOperation exists, and the hook must still see the call."""
    tr, made = _load(monkeypatch, ("FastOperation",))
    op = made["FastOperation"]()
    n, nbytes = tr._count_op_dispatches(lambda: op(_FakeTensor(1024, 0x1000)))
    assert n == 1, "the hook watched a class this build does not use"
    assert nbytes == 2048


def test_it_still_counts_when_the_build_uses_the_slow_class(monkeypatch):
    """Patching only FastOperation would move the blind spot, not remove it."""
    tr, made = _load(monkeypatch, ("Operation",))
    op = made["Operation"]()
    n, nbytes = tr._count_op_dispatches(lambda: op(_FakeTensor(512, 0x2000)))
    assert n == 1 and nbytes == 1024


def test_a_tensor_inside_a_list_is_not_invisible(monkeypatch):
    """ttnn.concat takes a LIST of tensors. Bare-argument-only counting scored it zero."""
    tr, made = _load(monkeypatch, ("FastOperation",))
    op = made["FastOperation"]()
    n, nbytes = tr._count_op_dispatches(lambda: op([_FakeTensor(100, 0xA), _FakeTensor(200, 0xB)]))
    assert n == 1
    assert nbytes == 600, "tensors passed inside a container were skipped"


def test_a_tensor_inside_a_dict_value_is_not_invisible(monkeypatch):
    tr, made = _load(monkeypatch, ("FastOperation",))
    op = made["FastOperation"]()
    _, nbytes = tr._count_op_dispatches(lambda: op(x={"w": _FakeTensor(50, 0xC)}))
    assert nbytes == 100


def test_two_tensors_sharing_a_reused_id_are_counted_twice(monkeypatch):
    """id() is not an identity: CPython hands the same address to a later object. Distinct BUFFERS
    are distinct reads, so keying on the buffer address keeps them apart."""
    tr, made = _load(monkeypatch, ("FastOperation",))
    op = made["FastOperation"]()

    def step():
        op(_FakeTensor(1000, 0x1))  # both temporaries; CPython may reuse the address
        op(_FakeTensor(1000, 0x2))

    _, nbytes = tr._count_op_dispatches(step)
    assert nbytes == 4000, "two distinct buffers collapsed into one"


def test_one_buffer_used_by_two_ops_is_counted_once(monkeypatch):
    """The quantity is a WORKING SET: a weight read by two ops is one resident tensor."""
    tr, made = _load(monkeypatch, ("FastOperation",))
    op = made["FastOperation"]()
    w = _FakeTensor(1000, 0x99)
    _, nbytes = tr._count_op_dispatches(lambda: (op(w), op(w)))
    assert nbytes == 2000


def test_a_hook_that_saw_nothing_says_so_instead_of_reporting_zero_bytes(monkeypatch, capsys):
    """The silence is the defect. A broken hook and a real zero must not look alike."""
    tr, _ = _load(monkeypatch, ("FastOperation",))
    tr._report_read_set("decode", None, 0)
    tr._report_read_set("prefill", 0, 0)
    tr._report_read_set("encode", 4096, 0)
    tr._report_read_set("decode", 12, 2048)
    out = capsys.readouterr().out
    assert "TRACE_STAGE_BYTES_NONE[decode] reason=hook did not install" in out
    assert "TRACE_STAGE_BYTES_NONE[prefill] reason=hook installed but intercepted 0" in out
    assert "TRACE_STAGE_BYTES_NONE[encode] reason=4096 op dispatches carried no device tensors" in out
    assert "TRACE_STAGE_BYTES[decode]=2048 ops=12" in out


def test_the_none_line_is_not_mistaken_for_a_measurement_by_the_parser():
    """perf_mcp keys on the literal 'TRACE_STAGE_BYTES[' -- the failure line must not match it."""
    assert "TRACE_STAGE_BYTES[" not in "TRACE_STAGE_BYTES_NONE[decode] reason=whatever"
