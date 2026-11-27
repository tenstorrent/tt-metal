# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf dir: GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor knobs (python literals) for
every perf test, so a whole guard run can be measured at a baseline setting (Refinement 11: e.g.
SPLIT_COST_MODEL=False,MIN_K_BLOCKS=1 reproduces the pre-refinement plans)."""
import ast
import os

import pytest


@pytest.fixture(autouse=True)
def _genify_matmul_knobs(monkeypatch):
    spec = os.environ.get("GENIFY_MATMUL_KNOBS", "")
    if spec:
        from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

        for item in filter(None, (s.strip() for s in spec.split(","))):
            name, value = item.split("=", 1)
            assert hasattr(pd, name), name
            monkeypatch.setattr(pd, name, ast.literal_eval(value))


@pytest.fixture(autouse=True)
def _genify_matmul_plan_trace(monkeypatch, request):
    """GENIFY_MATMUL_TRACE=1: print every R1 split the program descriptor chooses (grid, blocking, cost)."""
    if os.environ.get("GENIFY_MATMUL_TRACE") != "1":
        return
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    orig = pd.choose_split

    def _traced(geom, gx, gy, budget):
        grid, blk = orig(geom, gx, gy, budget)
        print(
            f"TRACE {request.node.name} mode={pd.operand_mode(geom, grid, blk)} "
            f"cost_us={pd.split_cost(geom, grid, blk) / 1350:.1f} {grid} {blk}"
        )
        return grid, blk

    monkeypatch.setattr(pd, "choose_split", _traced)
