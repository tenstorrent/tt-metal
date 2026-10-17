# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch: dump flat_routed_expert plans for PLAN_SHAPES ("H,I,E,NG,M;..." ) to PLAN_OUT_DIR/plan_H_I_E_M.json."""
import json
import os

import pytest

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_plan_any(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    for spec in os.environ["PLAN_SHAPES"].split(";"):
        H, I, E, NG, M = (int(v) for v in spec.split(","))
        p = dict(ttnn.bringup.flat_routed_expert_plan(mesh_device, H, I, E, NG, M, False, 1, False, False))
        p["grid"] = [g.x, g.y]
        out = f"{os.environ['PLAN_OUT_DIR']}/plan_{H}_{I}_{E}_{M}.json"
        json.dump(p, open(out, "w"), default=lambda o: list(o))
        print("PLAN_DUMP", spec, out)
