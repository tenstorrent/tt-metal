# SPDX-License-Identifier: Apache-2.0
"""Generic op-type-coverage probe (MODEL-AGNOSTIC).

Runs a perf test's forward at the current TT_PERF_LAYERS depth, intercepts EVERY dispatched ttnn op by
type (the same FastOperation-by-type technique the perf test uses to drain the profiler), and prints the
SET of distinct op signatures (op name + input-tensor shapes) as `PERF_OP_SIGS=<json>`. It wraps ttnn
itself and runs the given pytest node, so it needs no per-model knowledge — it works for any pipeline.

The coverage-window sizing (run.py:_coverage_layers) grows the profiled depth and compares these sets:
when a deeper window adds no new signature, every block type is covered and the profiled slice is a valid
representative sample. Homogeneous models saturate at 1-2 layers; heterogeneous ones (e.g. mamba + attention
+ MoE interleaved) grow until every type has appeared — with no model-specific layer maps.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent.parent.parent))
sys.path.insert(0, str(_PKG))

_SIGS = set()
_SEQ = []


def _shape_sig(args):
    out = []
    for x in args:
        s = getattr(x, "shape", None)
        if s is None:
            continue
        try:
            dims = tuple(int(d) for d in s)
        except Exception:  # noqa: BLE001
            dims = str(s)
        dt = getattr(getattr(x, "dtype", None), "name", None) or str(getattr(x, "dtype", "") or "")
        out.append((dims, dt) if dt else dims)
    return tuple(out)


def _wrap(fn, name):
    def inner(*a, **k):
        try:
            sig = "%s%s" % (name, _shape_sig(a))
            _SIGS.add(sig)
            _SEQ.append(sig)
        except Exception:  # noqa: BLE001
            pass
        return fn(*a, **k)

    return inner


def _install():
    import ttnn

    mods = [ttnn] + [getattr(ttnn, m, None) for m in ("transformer", "experimental")]
    for mod in [m for m in mods if m is not None]:
        for n in dir(mod):
            op = getattr(mod, n, None)
            if type(op).__name__ == "FastOperation":
                setattr(mod, n, _wrap(op, "%s.%s" % (getattr(mod, "__name__", "ttnn"), n)))


_BLOCK_TAG = "_perf_block_idx"
_SIGNPOST_PREFIX = "PERF_BLOCK_SIGNPOST:"


def _install_block_signposts():
    """Emit a real per-block signpost into the op stream at every repeated-block invocation, so a
    consumer can attribute each op to an exact block (not an inferred boundary). MODEL-AGNOSTIC: the
    largest nn.ModuleList in the built model is the repeated stack; its children are tagged with an
    index, and torch.nn.Module.__call__ is wrapped to drop a `PERF_BLOCK_SIGNPOST:<idx>` marker when a
    tagged block is entered. No per-model code, no markers baked into model source; probe-local only."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return

    orig = torch.nn.Module.__call__
    state = {"tagged": False}

    def _tag(root):
        best = None
        for m in root.modules():
            for _, child in m.named_children():
                if isinstance(child, torch.nn.ModuleList) and len(child) >= 2:
                    if best is None or len(child) > len(best):
                        best = child
        if best is None:
            return False
        for i, blk in enumerate(best):
            try:
                setattr(blk, _BLOCK_TAG, i)
            except Exception:  # noqa: BLE001
                pass
        return True

    def wrapped(self, *a, **k):
        if not state["tagged"]:
            try:
                if sum(1 for _ in self.modules()) > 8:
                    state["tagged"] = _tag(self)
            except Exception:  # noqa: BLE001
                pass
        idx = getattr(self, _BLOCK_TAG, None)
        if idx is not None:
            try:
                _SEQ.append("%s%d" % (_SIGNPOST_PREFIX, idx))
            except Exception:  # noqa: BLE001
                pass
        return orig(self, *a, **k)

    torch.nn.Module.__call__ = wrapped


def main(node: str, case: str | None = None) -> None:
    _install()
    _install_block_signposts()
    import pytest

    argv = ["-s", "-o", "timeout=0", node]
    if case:
        argv += ["-k", case]
    try:
        pytest.main(argv)
    except SystemExit:
        pass
    print("PERF_OP_SIGS=" + json.dumps(sorted(_SIGS)), flush=True)
    print("PERF_OP_SIG_COUNTS=" + json.dumps(Counter(_SEQ)), flush=True)
    print("PERF_OP_SIG_SEQUENCE=" + json.dumps(_SEQ[:50000]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
