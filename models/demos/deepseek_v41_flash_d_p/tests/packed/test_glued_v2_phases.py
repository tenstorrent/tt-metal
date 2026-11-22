# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7B diagnostic (not gated): the glued walls (tests/packed/test_glued_walls.py, its cases and env) with each profiled
chunk's device time split by profile phase printed (the walls record only the slowest chip's total). Run with the walls'
env and GLUED_WALL_CASES (e.g. 5x1600_p139264), GLUED_WALL_RUNS=2.

    BRINGUP_CHUNK=8192 <the gate's walls env> GLUED_WALL_CASES=5x1600_p139264 GLUED_WALL_RUNS=2 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_v2_phases.py
"""

from __future__ import annotations

import importlib

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_ITEM_VARS = ("GLUED_WALL_CASES", "PACKED_WALL_CAP", "GLUED_WALL_RUNS")
BATCH_LAYERS = "executed"


@mesh_parametrize(timeout_s=4 * 3600)
def test_glued_v2_phases(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tests.packed import test_glued_walls as G
    from models.demos.deepseek_v41_flash_d_p.tests.packed import test_packed_walls as W

    G = importlib.reload(G)
    orig = W._device_split

    def split(prof):
        out = orig(prof)
        ph = sorted(out["phases"].items(), key=lambda kv: -kv[1])
        print(f"PHASES slowest {out['slowest']:.2f} ms: " + ", ".join(f"{k} {v:.2f}" for k, v in ph[:16]), flush=True)
        return out

    monkeypatch.setattr(W, "_device_split", split)
    fn = getattr(G.test_glued_walls, "__wrapped__", G.test_glued_walls)
    fn(mesh_device, monkeypatch)
