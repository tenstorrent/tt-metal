# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Static check on the sweep definitions (host only, no ttnn import, no device).

A sweep that builds its mismatch gate as a one-element tuple, `allclose = (torch.allclose(...),)`
followed by `if not allclose:`, never rejects: a non-empty tuple is always truthy, so the vector is
accepted on PCC alone, and PCC does not change when the output is scaled by any positive factor.
"""

import ast
from pathlib import Path

import pytest

SWEEPS = Path(__file__).resolve().parent / "sweeps"
BOOLEAN_CHECK_NAMES = {"allclose"}


def _sweep_files():
    return sorted(SWEEPS.rglob("*.py"))


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))


def _tuple_gates(tree):
    """Return (variable, line) for gates assigned as a one-element tuple and used as `if not v`."""
    gates = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
            continue
        if len(node.value.elts) != 1 or not isinstance(node.value.elts[0], ast.Call):
            continue
        func = node.value.elts[0].func
        if not isinstance(func, ast.Attribute) or func.attr not in BOOLEAN_CHECK_NAMES:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                gates[target.id] = node.lineno

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name)
            and test.operand.id in gates
        ):
            hits.append((test.operand.id, gates[test.operand.id]))
    return hits


def test_allclose_gate_is_not_a_one_element_tuple():
    bad = []
    for path in _sweep_files():
        for _name, line in _tuple_gates(_parse(path)):
            bad.append(f"{path.relative_to(SWEEPS)}:{line}")
    assert not bad, (
        "these sweeps assign an allclose result as a one-element tuple; a non-empty tuple is always truthy, "
        f"so the mismatch branch is unreachable: {bad}"
    )
