# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device settings a model declares, read from its own test rather than invented.

OPENING A DEVICE IS NOT A DETAIL. l1_small_size feeds the scratch banks a model's convolution
front-end needs, and trace_region_size has to hold the largest traced stage; get either wrong and the
build fails or the trace does not fit. Every model states them -- pytest tests carry them as the
`device_params` parametrize, which is where the fixture gets them from.

WHY THIS EXISTS. A stack survey needs a BUILT model, and the first version got one by running a test
and hooking whatever it built. That made discovery depend on how a particular test happens to
construct the model, and it broke on the first test that did it differently: the correctness test ran
green and the hook never fired, so the survey reported "no stacks" for a model with two. Building the
model directly removes that dependency -- and the only thing standing in the way was these two
numbers, which the model has been declaring all along.

STATIC ONLY. This parses the test source; it never imports it. A test module pulls in ttnn, the model
package and its weights, so importing one to read two integers would cost more than the survey saves
and would fail in exactly the environments where discovery matters most.
"""

from __future__ import annotations

import ast
from pathlib import Path

DEFAULTS = {"l1_small_size": 24576}
_KEYS = ("l1_small_size", "trace_region_size", "dispatch_core_axis", "fabric_config", "num_command_queues")


def _module_constants(tree) -> dict:
    """Module-level NAME = <literal> bindings, so `{"l1_small_size": L1_SMALL_SIZE}` resolves."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if not targets:
                continue
            try:
                val = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                continue
            for t in targets:
                out[t.id] = val
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                pass
    return out


def _dicts_in(node, consts) -> list:
    """Every dict literal under `node`, with module constants substituted for bare names."""
    found = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Dict):
            continue
        row = {}
        ok = True
        for k, v in zip(sub.keys, sub.values):
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                ok = False
                break
            if isinstance(v, ast.Name) and v.id in consts:
                row[k.value] = consts[v.id]
            else:
                try:
                    row[k.value] = ast.literal_eval(v)
                except (ValueError, SyntaxError, TypeError):
                    ok = False
                    break
        if ok and row:
            found.append(row)
    return found


def from_source(src: str) -> dict:
    """The device_params a test file declares. {} when it declares none.

    Prefers a dict attached to a `device_params` parametrize -- that is the one the fixture consumes.
    Falls back to any module-level dict carrying device keys, which covers tests that build the
    params as a plain constant (the generated perf test does: _DEV_PARAMS = {...}).
    """
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return {}
    consts = _module_constants(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "parametrize" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and "device_params" in str(first.value):
            for row in _dicts_in(node, consts):
                if any(k in row for k in _KEYS):
                    return {k: v for k, v in row.items() if k in _KEYS}

    best = {}
    for node in tree.body:
        for row in _dicts_in(node, consts):
            picked = {k: v for k, v in row.items() if k in _KEYS}
            if len(picked) > len(best):
                best = picked
    return best


def for_model(model_root, prefer=None) -> dict:
    """Device params for a model, from `prefer` if given, else any test that declares them.

    Always returns something openable: a model that declares nothing still needs an l1_small_size,
    and the value below is the one the tool's own emitted skeleton uses.
    """
    root = Path(model_root)
    candidates = []
    if prefer:
        p = Path(prefer)
        candidates.append(p if p.is_absolute() else root / p)
    candidates += sorted(root.glob("tests/e2e/*.py")) + sorted(root.glob("tests/**/*.py"))
    for path in candidates:
        try:
            if not path.is_file():
                continue
            got = from_source(path.read_text())
        except OSError:
            continue
        if got:
            return got
    return dict(DEFAULTS)
