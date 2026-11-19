# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Static check on the sweep definitions (host only, no ttnn import, no device).

A sweep file named after one operator of a pair must not call or take its golden from the other
one, or the named operator is never run while the sweep reports green.
"""

import ast
from pathlib import Path

import pytest

SWEEPS = Path(__file__).resolve().parent / "sweeps"

# Pairs where the sweep file name commits it to one of the two operators.
OP_PAIRS = [("maximum", "minimum")]

# Names that are genuine aliases, so a golden under the other spelling is still the same op.
ALIASES = {
    "mul": "multiply",
    "sub": "subtract",
    "div": "divide",
    "floor_div": "floor_divide",
}


def _sweep_files():
    return sorted(SWEEPS.rglob("*.py"))


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))


def _called_names(tree, receiver="ttnn"):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == receiver:
                names.add(node.func.attr)
    return names


def _normalize(name):
    return ALIASES.get(name, name)


def test_sweep_runs_the_operator_named_in_the_file():
    bad = []
    for path in _sweep_files():
        tree = _parse(path)
        used = {_normalize(n) for n in _called_names(tree)}
        golden = {_normalize(n) for n in _named_ops_in_golden_lookup(tree)}
        for named, opposite in OP_PAIRS:
            if path.stem.startswith(named) and (opposite in used or opposite in golden):
                bad.append(f"{path.relative_to(SWEEPS)} is named after {named} but references {opposite}")
    assert not bad, f"these sweeps never run the operator they are named after: {bad}"


def _named_ops_in_golden_lookup(tree):
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "get_golden_function":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "ttnn":
                names.add(arg.attr)
    return names
