# SPDX-License-Identifier: Apache-2.0
"""Runtime native-ness probe.

A source-text check (regex OR an LLM reading the code) can be evaded by aliasing / obfuscation
(`_tt = ttnn.to_torch; _tt(x)`, `import torch as T; T.cat(...)`, operator forms like `x @ w`).
This probe instead measures what ACTUALLY EXECUTES during the stub's forward: it counts torch
compute ops (via a TorchFunctionMode that intercepts every torch API call, aliased or not) and ttnn
device dispatches (by wrapping FastOperation ops for the duration). A pure-ttnn forward dispatches
ttnn ops and runs ~zero torch ops; a host reimplementation runs torch ops regardless of how the
source is written. Un-evadable, deterministic.

The forward runs under `run_native_probe(...)`, which writes a sidecar next to the stub; the
graduation gate reads that sidecar as the authoritative verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

_TORCH_NONCOMPUTE = {
    "size",
    "dim",
    "numel",
    "shape",
    "is_floating_point",
    "is_complex",
    "is_tensor",
    "is_contiguous",
    "element_size",
    "get_default_dtype",
    "device",
    "dtype",
    "__get__",
    "__repr__",
    "__format__",
}


def probe_sidecar_path(stub_path) -> Path:
    return Path(str(stub_path) + ".native_probe.json")


def run_native_probe(stub_path, forward_thunk):
    """Execute forward_thunk() while counting torch compute ops and ttnn device dispatches; write the
    sidecar next to stub_path and return (forward_result, probe_dict). Never raises from the
    instrumentation itself — on any probe error it records torch_ops=-1 (unknown) so the gate falls
    back to its static check instead of trusting a bad reading."""
    counts = {"ttnn_dispatch": 0, "torch_ops": 0, "torch_op_names": set()}
    restore = []
    mode = None
    try:
        import ttnn

        for modname in ("", "transformer", "experimental"):
            mod = ttnn if modname == "" else getattr(ttnn, modname, None)
            if mod is None:
                continue
            for nm in dir(mod):
                op = getattr(mod, nm, None)
                if type(op).__name__ == "FastOperation":
                    restore.append((mod, nm, op))

                    def _wrap(orig):
                        def inner(*a, **k):
                            counts["ttnn_dispatch"] += 1
                            return orig(*a, **k)

                        return inner

                    setattr(mod, nm, _wrap(op))

        from torch.overrides import TorchFunctionMode

        class _Counter(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                name = getattr(func, "__name__", "") or str(func)
                if name not in _TORCH_NONCOMPUTE:
                    counts["torch_ops"] += 1
                    counts["torch_op_names"].add(name)
                return func(*args, **(kwargs or {}))

        mode = _Counter()
    except Exception:  # noqa: BLE001
        for mod, nm, op in restore:
            setattr(mod, nm, op)
        result = {"ttnn_dispatch": 0, "torch_ops": -1, "torch_op_names": []}
        _write_sidecar(stub_path, result)
        return forward_thunk(), result

    try:
        if mode is not None:
            with mode:
                out = forward_thunk()
        else:
            out = forward_thunk()
    finally:
        for mod, nm, op in restore:
            setattr(mod, nm, op)

    result = {
        "ttnn_dispatch": counts["ttnn_dispatch"],
        "torch_ops": counts["torch_ops"],
        "torch_op_names": sorted(counts["torch_op_names"])[:40],
    }
    _write_sidecar(stub_path, result)
    return out, result


def _write_sidecar(stub_path, result) -> None:
    try:
        probe_sidecar_path(stub_path).write_text(json.dumps(result, indent=2))
    except Exception:  # noqa: BLE001
        pass


def read_fresh_probe(stub_path):
    """The probe result for stub_path IF the sidecar exists and is at least as new as the stub (a
    stale probe from a previous stub version is ignored). Returns the dict or None."""
    stub = Path(stub_path)
    side = probe_sidecar_path(stub)
    try:
        if not side.is_file() or not stub.is_file():
            return None
        if side.stat().st_mtime < stub.stat().st_mtime:
            return None
        return json.loads(side.read_text())
    except Exception:  # noqa: BLE001
        return None


def is_native_from_probe(result, max_torch_ops: int = 0) -> bool | None:
    """True/False from a runtime probe, or None when the probe is unusable (torch_ops=-1) so the
    caller keeps its static check. Native = dispatched ttnn device ops AND ran <= max_torch_ops torch
    compute ops."""
    if not result or result.get("torch_ops", -1) < 0:
        return None
    return result.get("ttnn_dispatch", 0) > 0 and result.get("torch_ops", 0) <= max_torch_ops
