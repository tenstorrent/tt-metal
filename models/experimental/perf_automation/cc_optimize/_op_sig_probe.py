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
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent.parent.parent))
sys.path.insert(0, str(_PKG))

_SIGS = set()


def _shape_sig(args):
    out = []
    for x in args:
        s = getattr(x, "shape", None)
        if s is None:
            continue
        try:
            out.append(tuple(int(d) for d in s))
        except Exception:  # noqa: BLE001
            out.append(str(s))
    return tuple(out)


def _wrap(fn, name):
    def inner(*a, **k):
        try:
            _SIGS.add("%s%s" % (name, _shape_sig(a)))
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


def main(node: str, case: str | None = None) -> None:
    _install()
    import pytest

    argv = ["-s", "-o", "timeout=0", node]
    if case:
        argv += ["-k", case]
    try:
        pytest.main(argv)
    except SystemExit:
        pass
    print("PERF_OP_SIGS=" + json.dumps(sorted(_SIGS)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
