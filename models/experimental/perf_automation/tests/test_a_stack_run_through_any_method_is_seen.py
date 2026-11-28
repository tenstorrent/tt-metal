# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A block stack is seen whichever method its container runs it through.

GLM-5.3-Flash, 2026-11-09. The vision tower runs its blocks as ``blk(x)``; the text model runs its 45
layers as ``layer.forward(X, step)``. The signpost bracket sat on ``__call__`` only, so the text
layers were tagged and silent: coverage saw one stack (vision, saturating at 2), applied that 2 to
the text stack too, and profiled two linear-attention/dense layers -- never reaching layer 3, where
sparse attention and the MoE first appear.

These run on CPU with no device. The blocks are plain classes (no framework base), the containers
are plain lists, and the methods are named arbitrarily, so nothing here passes because of a name.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent import caller_stacks as cs  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(_PA / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model(seq, n_called=4, n_method=6, method_name="advance"):
    """Two stacks, two call shapes, both GLM's: blocks that own sub-modules and define __call__, one
    stack run as ``blk(x)`` and the other through a METHOD its container calls directly. Each block
    appends op tokens; from block 3 the method-driven stack appends a NEW op, so it needs 4 blocks."""

    class Sub:
        def __call__(self, x):
            return x

    class CalledBlock:
        def __init__(self):
            self.sub = Sub()

        def __call__(self, x):
            seq.append("op_called")
            return x

    def _run(self, x):
        seq.append("op_common")
        if self.kind:
            seq.append("op_deep_kind")
        return self.helper(x)

    def _helper(self, x):  # a public method the entry point calls: must NOT open a second frame
        seq.append("op_helper")
        return x

    def _call(self, x):  # defined, as GLM's is, but NOT how the container runs the stack
        return getattr(self, method_name)(x)

    MethodBlock = type("MethodBlock", (), {method_name: _run, "helper": _helper, "__call__": _call})

    class Tower:
        def __init__(self):
            self.blocks = [CalledBlock() for _ in range(n_called)]

        def __call__(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    class Stack:
        def __init__(self):
            self.layers = []
            for i in range(n_method):
                m = MethodBlock()
                m.sub = Sub()
                m.kind = i >= 3
                self.layers.append(m)

        def run(self, x):
            for layer in self.layers:
                x = getattr(layer, method_name)(x)
            return x

    class Pipeline:
        def __init__(self):
            self.tower = Tower()
            self.stack = Stack()

        def go(self, x):
            return self.stack.run(self.tower(x))

    return Pipeline()


# ------------------------------------------------------------------ the shared helper


def test_entry_points_are_read_off_the_class_not_assumed():
    class A:
        def forward(self):
            pass

        def advance(self):
            pass

        def _private(self):
            pass

        @staticmethod
        def s():
            pass

        @classmethod
        def c(cls):
            pass

        @property
        def p(self):
            return 1

    assert sorted(cs.entry_points(A)) == ["advance", "forward"]

    class B(A):
        def __call__(self):
            pass

    assert cs.entry_points(B)[0] == "__call__" and set(cs.entry_points(B)) == {"__call__", "advance", "forward"}


def test_torch_modules_keep_call_only():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def forward(self, x):
            return x

        def extra(self):
            pass

    assert cs.entry_points(M) == ["__call__"]


def test_nested_entry_points_open_one_frame():
    events = []

    class L:
        def __call__(self, x):
            return self.forward(x)

        def forward(self, x):
            return self.helper(x)

        def helper(self, x):
            return x

    assert cs.bracket_entry_points(L, lambda s: events.append("in"), lambda s: events.append("out"), "_k")
    obj = L()
    obj(1)
    obj.forward(1)
    obj.helper(1)
    assert events == ["in", "out"] * 3


def test_the_frame_closes_when_the_block_raises():
    events = []

    class L:
        def forward(self):
            raise RuntimeError("boom")

    cs.bracket_entry_points(L, lambda s: events.append("in"), lambda s: events.append("out"), "_k")
    obj = L()
    for _ in range(2):  # the guard resets: a second call still brackets
        with pytest.raises(RuntimeError):  # allow-pytest.raises: no expect_error fixture
            obj.forward()
    assert events == ["in", "out", "in", "out"]


def test_two_brackets_on_one_class_do_not_suppress_each_other():
    a, b = [], []

    class L:
        def forward(self):
            pass

    cs.bracket_entry_points(L, lambda s: a.append("in"), lambda s: a.append("out"), "_first")
    cs.bracket_entry_points(L, lambda s: b.append("in"), lambda s: b.append("out"), "_second")
    L().forward()
    assert a == ["in", "out"] and b == ["in", "out"]


def test_wrapping_is_once_per_class_per_key():
    events = []

    class L:
        def forward(self):
            pass

    assert cs.bracket_entry_points(L, lambda s: events.append("in"), lambda s: events.append("out"), "_k")
    assert not cs.bracket_entry_points(L, lambda s: events.append("in"), lambda s: events.append("out"), "_k")
    L().forward()
    assert events == ["in", "out"]


# ------------------------------------------------------------------ the caller witness


def test_the_caller_witness_sees_a_method_driven_stack():
    seq = []
    p = _model(seq)
    assert cs.instrument(p, seq.append) > 0
    p.go(0)
    paths = {s["path"]: s for s in cs.observed_stacks(seq)}
    assert any(s["ran"] == 6 for s in paths.values()), "the method-driven stack never ran: %s" % paths
    assert any(s["ran"] == 4 for s in paths.values()), "the __call__-driven stack never ran: %s" % paths


# ------------------------------------------------------------------ the block signposts, end to end


@pytest.mark.parametrize("method_name", ["forward", "advance", "step_once"])
def test_both_stacks_signpost_and_the_deep_kind_sizes_the_window(method_name):
    probe = _load("_op_sig_probe_any_method", "cc_optimize/_op_sig_probe.py")
    run = _load("cc_run_any_method", "cc_optimize/run.py")
    try:
        from models.common.lightweightmodule import LightweightModule

        lw_orig = LightweightModule.__call__
    except Exception:  # noqa: BLE001
        LightweightModule, lw_orig = None, None
    try:
        probe._SEQ.clear()
        probe._install_block_signposts()
        p = _model(probe._SEQ, method_name=method_name)
        assert probe.tag_built_model(p)
        p.go(0)
        seq = list(probe._SEQ)
    finally:
        if LightweightModule is not None:
            LightweightModule.__call__ = lw_orig

    stacks = run._stack_ids_from_seq(seq)
    assert len(stacks) == 2, "expected both stacks to signpost, saw %s" % stacks
    fb, source = run._first_block_map(seq)
    assert source == "signposts"
    deepest = {sid: max(m.values()) for sid, m in fb.items()}
    assert sorted(deepest.values()) == [0, 3], deepest
    # the helper the entry point calls is attributed to the same block, not a second frame
    method_sid = next(sid for sid, d in deepest.items() if d == 3)
    assert fb[method_sid]["op_helper"] == 0 and fb[method_sid]["op_deep_kind"] == 3
