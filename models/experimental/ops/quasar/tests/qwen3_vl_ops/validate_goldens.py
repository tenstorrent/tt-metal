# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Offline check that every GOLDEN reference runs against its cases — no ttnn, no device.

    python models/experimental/ops/quasar/tests/qwen3_vl_ops/validate_goldens.py

``validate_cases.py`` checks the generated *data* (ids unique, enums known, program-config
fields mapped). This checks the *reference code* on the other side: for each case whose op
has an entry in ``graph_case.GOLDEN``, build torch inputs shaped like that case and call
the reference.

It exists because a reference that crashes on one case's argument shape is invisible until
a device run reaches it -- which is how ``ttnn.multiply(cache, 0, output_tensor=cache)``
(a scalar right-hand operand, so ``inputs["1"]`` does not exist) and ``_ref_split`` reading
a literal spec dict where it wanted the literal's value both got through review.

What it does NOT check: that the reference computes the *right* answer. Only a device run
comparing against it can say that. A reference returning None is reported too -- that is
the "no reference applies, fall back to structural checks" path, which is legitimate but
worth seeing, since a reference that silently returns None for every case looks like it
passes while checking nothing.

graph_case imports ttnn, so it is never imported here: its module-level defs are exec'd
individually and the ones that need ttnn (the enum tables) are skipped.
"""

from __future__ import annotations

import ast
import math
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
_INT_DTYPES = {"UINT32", "INT32", "UINT16", "UINT8"}


def _load_goldens(path: pathlib.Path):
    """(GOLDEN table, namespace) from graph_case.py source, without importing ttnn."""
    tree = ast.parse(path.read_text())
    ns: dict = {"torch": torch, "math": math}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            try:
                exec(compile(ast.Module([node], []), str(path), "exec"), ns)
            except Exception:
                pass  # needs ttnn (the enum tables); no reference depends on them
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "GOLDEN":
            # Values are expressions: a bare name, or _ref_binary(torch.add).
            return {ast.literal_eval(k): eval(ast.unparse(v), ns) for k, v in zip(node.value.keys, node.value.values)}
    raise SystemExit(f"{path}: no GOLDEN table found")


def _data(spec):
    """Mirror graph_case._torch_data's dtype split (int tensors are indices -> zeros)."""
    if spec["dtype"] in _INT_DTYPES:
        return torch.zeros(spec["shape"], dtype=torch.int32)
    return torch.randn(*spec["shape"]).to(torch.bfloat16)


def _inputs(case):
    """Mirror the keys graph_case's torch_sink ends up with, list elements included."""
    out = {}

    def add(key, spec):
        if spec.get("k") == "t":
            out[key] = _data(spec)
        elif spec.get("k") == "tlist":
            for j, sub in enumerate(spec["tensors"]):
                out[f"{key}[{j}]"] = _data(sub)

    for i, spec in enumerate(case["args"]):
        add(str(i), spec)
    for name, spec in case["kwargs"].items():
        add(name, spec)
    return out


def _cases(path: pathlib.Path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "CASES":
            return ast.literal_eval(node.value)
    return []


def main():
    torch.manual_seed(0)
    golden = _load_goldens(HERE / "graph_case.py")
    errors, notes = [], []
    ran = 0

    for path in sorted(HERE.glob("test_*.py")):
        for case in _cases(path):
            ref_fn = golden.get(case["op"])
            if ref_fn is None:
                continue
            try:
                result = ref_fn(_inputs(case), {}, case)
            except Exception as exc:  # a reference must never raise: that is a crash on device
                errors.append(f"{path.name}:{case['id']}: {type(exc).__name__}: {exc}")
                continue
            ran += 1
            if result is None:
                notes.append(f"{path.name}:{case['id']}: reference returned None (structural checks only)")

    print(f"{len(golden)} reference(s) wired, {ran} case(s) exercised\n")
    for note in notes:
        print(f"  note  {note}")
    for err in errors:
        print(f"  ERROR {err}")
    print(f"\n{len(errors)} error(s), {len(notes)} note(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
