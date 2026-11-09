# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test of the root conftest's QWEN36_TEST_FULL_MESH planner (no device, no ttnn build: the function is
extracted from conftest.py with ast and run against a stub ttnn)."""
import ast
import os
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[5]


def _plan_fn():
    src = (_ROOT / "conftest.py").read_text()
    node = next(
        n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_full_mesh_submesh_plan"
    )
    stub = types.ModuleType("ttnn")
    stub.FabricConfig = types.SimpleNamespace(FABRIC_1D="1D", FABRIC_1D_RING="RING", FABRIC_2D_TORUS_XY="TORUS")
    ns = {"os": os}
    sys.modules.setdefault("ttnn", stub)
    saved = sys.modules["ttnn"]
    sys.modules["ttnn"] = stub
    try:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "conftest.py", "exec"), ns)
    finally:
        sys.modules["ttnn"] = saved
    fn = ns["_full_mesh_submesh_plan"]

    def call(*a, **k):
        sys.modules["ttnn"] = stub
        try:
            return fn(*a, **k)
        finally:
            sys.modules["ttnn"] = saved

    return call


def test_off_by_default():
    plan = _plan_fn()
    assert plan((1, 8), 32, env={}) is None
    assert plan((1, 8), 32, env={"QWEN36_TEST_FULL_MESH": "0"}) is None


@pytest.mark.parametrize("cols", [1, 2, 4, 8])
def test_on_galaxy_rows(cols):
    plan = _plan_fn()
    assert plan((1, cols), 32, env={"QWEN36_TEST_FULL_MESH": "1"}) == ((4, 8), (1, cols), 0, "RING")


def test_submesh_index_and_fabric_override():
    plan = _plan_fn()
    env = {"QWEN36_TEST_FULL_MESH": "1", "QWEN36_TEST_SUBMESH": "3", "QWEN36_TEST_FULL_MESH_FABRIC": "FABRIC_2D_TORUS_XY"}
    assert plan((1, 8), 32, env=env) == ((4, 8), (1, 8), 3, "TORUS")


@pytest.mark.parametrize("shape,size", [((1, 8), 8), ((2, 8), 32), ((1, 16), 32), ((4, 8), 32), ((1, 3), 32)])
def test_not_applicable(shape, size):
    plan = _plan_fn()
    assert plan(shape, size, env={"QWEN36_TEST_FULL_MESH": "1"}) is None
