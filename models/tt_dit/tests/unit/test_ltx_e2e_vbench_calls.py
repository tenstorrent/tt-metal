# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU: every call of the LTX e2e test's nested VBench helpers passes all their required arguments.

The helpers only run after a full device generation, so a call that misses an argument (main changed
check_output_with_vbench to (prompt, number, seed, ...) while ltx-rt's replay loop still passed two)
fails only on the device, after the whole run.
"""

import ast
from pathlib import Path

import pytest

E2E = Path(__file__).resolve().parents[1] / "models/ltx/test_pipeline_ltx_distilled.py"


def _nested_defs_and_calls():
    tree = ast.parse(E2E.read_text())
    defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("check_output_with_"):
            args = node.args
            required = len(args.posonlyargs) + len(args.args) - len(args.defaults)
            defs[node.name] = (required, [a.arg for a in args.posonlyargs + args.args])
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in defs
    ]
    return defs, calls


def test_helpers_are_found():
    defs, calls = _nested_defs_and_calls()
    assert "check_output_with_vbench" in defs
    assert calls


@pytest.mark.parametrize("call", _nested_defs_and_calls()[1], ids=lambda c: f"line{c.lineno}")
def test_call_passes_required_arguments(call):
    defs, _ = _nested_defs_and_calls()
    required, names = defs[call.func.id]
    given = set(names[: len(call.args)]) | {k.arg for k in call.keywords}
    missing = [n for n in names[:required] if n not in given]
    assert not missing, f"{call.func.id} at {E2E.name}:{call.lineno} misses {missing}"
