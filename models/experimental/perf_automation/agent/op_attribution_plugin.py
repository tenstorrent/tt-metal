"""op->source attribution pytest plugin (the "eyes").

Wraps the dominant ttnn ops (matmul / linear / shard movers) to record the
model-source line that issued each one, into TT_OP_ATTR_OUT. op_attribution.aggregate
then ranks source lines by matmul work, turning "a slow matmul somewhere" into
"decoder_layer.py:382 emits the dominant matmuls". Keyed off TT_OP_ATTR_ROOT, so it
works for any model. Mirrors exec_trace_plugin's lifecycle."""

from __future__ import annotations

import json
import os
import traceback

_RECORDS: list[dict] = []
_ROOT = ""  # only attribute frames under this path (the model dir / repo)


def _model_frame(root: str):
    """The deepest stack frame in the MODEL SOURCE that issued the op.

    Must skip (a) the perf-test harness under `<model>/tests/` — the test itself
    wraps ttnn.linear/matmul for the profiler flush, so its wrapper frame would
    otherwise win and mis-attribute every op to test_*.py — and (b) the tool's own
    code. Target the real model implementation (_stubs/ or tt/); the deepest such
    frame is the line that actually called the ttnn op (e.g. decoder_layer.py:382)."""
    best = None
    for fr in traceback.extract_stack()[:-2]:  # drop our wrapper frames
        f = fr.filename
        if not root or root not in f:
            continue
        if "/tests/" in f or "/perf_automation/" in f:  # skip test harness + the tool
            continue
        if "/_stubs/" in f or "/tt/" in f:  # the actual model implementation
            best = fr  # DEEPEST model-source frame = the op's true call site
    return best


def _wrap(fn, op_name):
    def inner(*args, **kwargs):
        try:
            fr = _model_frame(_ROOT)
            if fr is not None:
                shape = None
                try:
                    shape = tuple(args[0].shape) if args and hasattr(args[0], "shape") else None
                except Exception:
                    shape = None
                _RECORDS.append(
                    {
                        "seq": len(_RECORDS),
                        "op": op_name,
                        "src": f"{fr.filename}:{fr.lineno}",
                        "func": fr.name,
                        "shape": shape,
                    }
                )
        except Exception:
            pass  # attribution must never break the workload
        return fn(*args, **kwargs)

    return inner


def pytest_configure(config):
    global _ROOT
    _ROOT = os.environ.get("TT_OP_ATTR_ROOT", "")
    try:
        import ttnn
    except Exception:
        return
    # Wrap the dominant compute + the data-movement ops a shard would add/remove,
    # so attribution covers both "where the matmul is" and "where a reshard would land".
    for name in ("matmul", "linear", "to_memory_config", "sharded_to_interleaved", "interleaved_to_sharded"):
        fn = getattr(ttnn, name, None)
        if fn is not None and not getattr(fn, "_tt_attr_wrapped", False):
            w = _wrap(fn, name)
            w._tt_attr_wrapped = True
            setattr(ttnn, name, w)


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("TT_OP_ATTR_OUT")
    if not out:
        return
    try:
        with open(out, "w") as f:
            for r in _RECORDS:
                f.write(json.dumps(r) + "\n")
    except Exception:
        pass
