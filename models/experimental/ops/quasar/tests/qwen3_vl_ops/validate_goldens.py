# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Check that every GOLDEN reference runs against its cases — no device needed.

    python models/experimental/ops/quasar/tests/qwen3_vl_ops/validate_goldens.py

``validate_cases.py`` checks the generated *data* (ids unique, enums known, program-config
fields mapped). This checks the *reference code* on the other side: for each case whose op
has an entry in ``graph_case.GOLDEN``, build torch inputs shaped like that case and call
the reference.

It exists because a reference that crashes on one case's argument shape is invisible until
a device run reaches it — which is how ``ttnn.multiply(cache, 0, output_tensor=cache)``
(a scalar right-hand operand, so ``inputs["1"]`` does not exist) and ``_ref_split`` reading
a literal spec dict where it wanted the literal's value both got through review.

What it does NOT check: that a reference computes the *right* answer. Only a device run
comparing against it can say that. A reference returning None is reported too — that is
the "no reference applies, fall back to structural checks" path, which is legitimate but
worth seeing, since a reference that silently returns None for every case looks like it
passes while checking nothing.

This imports ``graph_case`` and the generated test modules the ordinary way, so it needs
``ttnn`` importable (no device is opened, nothing is dispatched). Modules are imported by
dotted path rather than read as text on purpose: reading a module and exec'ing it would
be the same work with none of the import system's guarantees.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
_INT_DTYPES = {"UINT32", "INT32", "UINT16", "UINT8"}


def _package() -> str:
    """Dotted path of this directory, so the generated modules import as themselves."""
    root = next((p for p in HERE.parents if (p / ".git").exists()), None)
    if root is None:
        raise SystemExit(f"cannot locate the repo root above {HERE}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return ".".join(HERE.relative_to(root).parts)


def _data(spec):
    """Mirror graph_case._torch_data's dtype split (int tensors are indices -> zeros)."""
    if spec["dtype"] in _INT_DTYPES:
        return torch.zeros(spec["shape"], dtype=torch.int32)
    return torch.randn(*spec["shape"]).to(torch.bfloat16)


def _inputs(case):
    """Mirror the keys graph_case's torch_sink ends up with, list elements included."""
    built = {}

    def add(key, spec):
        if spec.get("k") == "t":
            built[key] = _data(spec)
        elif spec.get("k") == "tlist":
            for i, sub in enumerate(spec["tensors"]):
                built[f"{key}[{i}]"] = _data(sub)

    for index, spec in enumerate(case["args"]):
        add(str(index), spec)
    for name, spec in case["kwargs"].items():
        add(name, spec)
    return built


def main():
    torch.manual_seed(0)
    package = _package()
    golden = importlib.import_module(f"{package}.graph_case").GOLDEN

    errors, notes = [], []
    ran = 0
    for path in sorted(HERE.glob("test_*.py")):
        cases = getattr(importlib.import_module(f"{package}.{path.stem}"), "CASES", [])
        for case in cases:
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
