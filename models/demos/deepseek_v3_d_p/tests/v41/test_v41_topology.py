# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 collective topology per mesh axis follows the opened fabric (bead 8y7.13.6, no device).

LoudBox opens FABRIC_2D (no wrapped axis): Linear on SP and TP, as before. The Galaxy torus-xy-8x4 / 4x8 profile
opens FABRIC_2D_TORUS_XY: Ring on both axes. A fabric wrapping one axis rings only that axis. The source scan
guards the contract that no V4.1 module picks a topology itself."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41 import ccl as v41_ccl
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives

L, R = ttnn.Topology.Linear, ttnn.Topology.Ring
FABRICS = {
    "loudbox-fabric2d": (ttnn.FabricConfig.FABRIC_2D, (L, L)),
    "galaxy-torus-xy": (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (R, R)),
    "torus-x": (ttnn.FabricConfig.FABRIC_2D_TORUS_X, (L, R)),  # X = mesh axis 1 = TP
    "torus-y": (ttnn.FabricConfig.FABRIC_2D_TORUS_Y, (R, L)),  # Y = mesh axis 0 = SP
    "disabled": (ttnn.FabricConfig.DISABLED, (L, L)),
}


@pytest.mark.parametrize("fabric, expected", list(FABRICS.values()), ids=list(FABRICS))
def test_collectives_take_axis_topology_from_fabric(fabric, expected, monkeypatch):
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: fabric)
    monkeypatch.setattr(v41_ccl, "is_blackhole", lambda: True)
    ccl = V41Collectives(SimpleNamespace(shape=(1, 1)))  # 1x1: no tt_ccl, so no device is touched
    assert (ccl.sp_topology, ccl.tp_topology) == expected


def test_no_v41_module_hardcodes_a_topology():
    root = Path(v41_ccl.__file__).parent
    hits = [
        f"{p.name}:{i}: {line.strip()}"
        for p in sorted(root.glob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"Topology\.(Linear|Ring)", line)
    ]
    assert not hits, "V4.1 topology must come from the fabric (tt_ccl.per_axis_topology):\n" + "\n".join(hits)
