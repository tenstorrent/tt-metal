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


def _largest_repeated_stack(root, _depth: int = 0, _seen=None):
    """The largest list/tuple of SAME-TYPED objects reachable from `root` — i.e. the repeated block
    stack of a model that is not built out of torch containers.

    A TTNN model is typically NOT a torch.nn.Module: models/common/lightweightmodule.py exists
    precisely to avoid torch's per-call host overhead, and such models hold their decoder blocks in a
    PLAIN PYTHON LIST (``self.layers = [TransformerBlock(...) for _ in range(n_layers)]``). Looking
    only for nn.ModuleList therefore finds nothing on most tt-metal models, the probe emits no
    signposts, and run.py has to fall back to probing depth 2/4/8/16 to discover what the signposts
    would have said for free.

    Same-typedness is the signal, not the attribute name: a stack is N instances of one class, so no
    per-model knowledge (and no 'layers'/'blocks' name list) is needed.
    """
    if _seen is None:
        _seen = set()
    if root is None or _depth > 4 or id(root) in _seen:
        return None
    _seen.add(id(root))
    best = None
    for value in list(getattr(root, "__dict__", {}).values()):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            kinds = {type(v) for v in value if v is not None and hasattr(v, "__dict__")}
            if len(kinds) == 1 and (best is None or len(value) > len(best)):
                best = list(value)
        elif hasattr(value, "__dict__"):
            deeper = _largest_repeated_stack(value, _depth + 1, _seen)
            if deeper is not None and (best is None or len(deeper) > len(best)):
                best = deeper
    return best


def _tag_stack(stack) -> bool:
    """Index every block in `stack` so entering one can be attributed to an exact depth."""
    if not stack:
        return False
    tagged = False
    for i, blk in enumerate(stack):
        try:
            setattr(blk, _BLOCK_TAG, i)
            tagged = True
        except Exception:  # noqa: BLE001
            pass
    return tagged


def _install_block_signposts():
    """Emit a real per-block signpost into the op stream at every repeated-block invocation, so a
    consumer can attribute each op to an exact block (not an inferred boundary).

    MODEL-AGNOSTIC, and it must cover BOTH shapes tt-metal models come in:

      torch-shaped      the largest nn.ModuleList is the stack; torch.nn.Module.__call__ is wrapped.
      TTNN-shaped       blocks subclass LightweightModule (NOT torch.nn.Module) and live in a plain
                        Python list, so the torch hook never fires for them. LightweightModule.__call__
                        is wrapped too, and the stack is found by looking for the largest list of
                        same-typed objects.

    Covering only the torch shape is why llama3_1_8b_p150 reported full_blocks=0 and run.py had to
    climb a 2/4/8/16 ladder — four extra device probes — to recover depths this would have supplied
    from the single all-layers probe.

    No per-model code, no markers baked into model source; probe-local only.
    """
    state = {"tagged": False}

    def _emit(self):
        idx = getattr(self, _BLOCK_TAG, None)
        if idx is not None:
            try:
                _SEQ.append("%s%d" % (_SIGNPOST_PREFIX, idx))
            except Exception:  # noqa: BLE001
                pass

    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None

    if torch is not None:
        _torch_orig = torch.nn.Module.__call__

        def _torch_tag(root):
            best = None
            for m in root.modules():
                for _, child in m.named_children():
                    if isinstance(child, torch.nn.ModuleList) and len(child) >= 2:
                        if best is None or len(child) > len(best):
                            best = child
            if best is None:
                return False
            return _tag_stack(list(best))

        def _torch_wrapped(self, *a, **k):
            if not state["tagged"]:
                try:
                    if sum(1 for _ in self.modules()) > 8:
                        state["tagged"] = _torch_tag(self)
                except Exception:  # noqa: BLE001
                    pass
            _emit(self)
            return _torch_orig(self, *a, **k)

        torch.nn.Module.__call__ = _torch_wrapped

    try:
        from models.common.lightweightmodule import LightweightModule
    except Exception:  # noqa: BLE001
        return

    _lw_orig = LightweightModule.__call__

    def _lw_wrapped(self, *a, **k):
        if not state["tagged"]:
            try:
                state["tagged"] = _tag_stack(_largest_repeated_stack(self))
            except Exception:  # noqa: BLE001
                pass
        _emit(self)
        return _lw_orig(self, *a, **k)

    LightweightModule.__call__ = _lw_wrapped


def main(node: str, case: str | None = None) -> None:
    _install()
    _install_block_signposts()
    import pytest

    # The probe asks for ALL layers (the caller removed the cap); load the depth guard so a
    # setdefault in the test module cannot quietly reinstate one before the model is built.
    argv = ["-s", "-p", "models.experimental.perf_automation.agent.depth_guard_plugin", "-o", "timeout=0", node]
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
