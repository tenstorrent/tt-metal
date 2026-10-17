# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch: dump the flat_routed_expert plan at GLM-5.3-Flash's model shape (one chip) as JSON."""
import json
import os

import pytest

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_plan_dump(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    p = ttnn.bringup.flat_routed_expert_plan(mesh_device, 4096, 2048, 36, 288, 8192, False, 1, False, False)
    d = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(p).items()}
    d["grid"] = [g.x, g.y]
    out = os.environ["PLAN_OUT"]
    json.dump(d, open(out, "w"), default=lambda o: list(o))
    print("PLAN_DUMP", out, {k: d[k] for k in ("np", "g", "mt", "nd", "n_rd", "hbuf", "rdown", "n_rdn", "pcd_r", "pcds")})
