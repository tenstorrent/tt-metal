# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for the per-axis CCL topology mapping.

`per_axis_topology()` maps a FabricConfig to the (sp_topology, tp_topology) tuple that drives every
prefill collective's `topology=`. A wrong entry silently rings (or fails to ring) the wrong axis and,
on a torus fabric, deadlocks on a missing wrap link — but that only surfaces on a wrap-cabled galaxy
(the device tests are skipped in CI). These no-device tests pin the mapping so an accidental axis
inversion is caught in ordinary CI instead of a hardware run.
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

_LINEAR = ttnn.Topology.Linear
_RING = ttnn.Topology.Ring


@pytest.mark.parametrize(
    "fabric_config, expected",
    [
        # X = cols = dim 1 = tp_axis; Y = rows = dim 0 = sp_axis. (sp_topology, tp_topology).
        (ttnn.FabricConfig.FABRIC_2D_TORUS_X, (_LINEAR, _RING)),  # X wrap -> TP rings
        (ttnn.FabricConfig.FABRIC_2D_TORUS_Y, (_RING, _LINEAR)),  # Y wrap -> SP rings
        (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (_RING, _RING)),  # both wrapped
        (ttnn.FabricConfig.FABRIC_1D_RING, (_RING, _LINEAR)),  # 1D ring on the SP axis
        (ttnn.FabricConfig.FABRIC_2D, (_LINEAR, _LINEAR)),  # non-torus -> no ring
        (ttnn.FabricConfig.FABRIC_1D, (_LINEAR, _LINEAR)),
    ],
)
def test_per_axis_topology(fabric_config, expected):
    assert per_axis_topology(fabric_config) == expected


def test_per_axis_topology_disabled_fabric_is_linear():
    # Any unmapped/non-ring fabric must degrade to all-Linear (never a spurious Ring).
    assert per_axis_topology(ttnn.FabricConfig.DISABLED) == (_LINEAR, _LINEAR)


# Fabrics whose block-test params carry an explicit per-axis (sp, tp) tuple.
_RING_TORUS_FABRICS = {
    ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricConfig.FABRIC_1D_RING,
}


def test_conftest_torus_params_match_helper():
    """Every torus block-test param's hardcoded (sp, tp) tuple must equal per_axis_topology(fabric).

    This links the conftest params to the single source of truth so the two can't silently drift:
    if the helper mapping is corrected, this fails until the params are updated to match (and vice
    versa). Param layout: pytest.param(mesh_shape, device_params, num_links, topology, ...).
    """
    from models.demos.deepseek_v3_d_p.tests.conftest import FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS

    checked = 0
    for param in FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS:
        values = param.values
        device_params = values[1] if len(values) > 1 and isinstance(values[1], dict) else {}
        fabric = device_params.get("fabric_config")
        if fabric not in _RING_TORUS_FABRICS:
            continue
        topology = values[3]
        assert topology == per_axis_topology(fabric), (
            f"{param.id}: param topology {topology} != per_axis_topology({fabric}) "
            f"{per_axis_topology(fabric)} — update the param or the mapping so they agree."
        )
        checked += 1
    assert checked >= 5, f"expected to validate the torus params, only matched {checked}"
