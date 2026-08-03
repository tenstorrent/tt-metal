# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared mesh configuration parameters for dispatch/combine PCC tests.

The op_unit_tests test_prefill_dispatch.py, test_ttnn_dispatch_combine.py, and
test_combine_subdevices.py, plus perf/test_prefill_dispatch_combine.py, import
ALL_MESH_CONFIGS to avoid duplicating the same pytest.param entries.
test_combine_subdevices.py pins the `fabric2d-mesh-4x2` ID.

FabricConfig is the single source of truth. Consumers derive their cluster-axis CCL
topology with ``per_axis_topology`` instead of carrying a second parameter.

Test-id naming convention
-------------------------
- Local IDs are `fabric2d-mesh-2x2`, `fabric2d-mesh-2x4`, the existing
  one-link/two-link 4x2 diagnostics, and TorusY Nx1 proxies.
- Production IDs are `fabric2d-torus-xy-8x4-{1,2}link`.
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    fabric_1d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_max_payload_size


<<<<<<< HEAD
def _mesh_param(shape, fabric, payload, nlinks, topo_marker, test_id, reliability_mode=None):
=======
def _mesh_param(shape, fabric, payload, nlinks, topo, topo_marker, test_id, reliability_mode=None):
>>>>>>> fac3bf836d1 (switching to leaner test changes)
    """Build a single pytest.param for the mesh_device parametrize axis.

    `topo_marker` is the CI hardware-class string consumed by the `requires_mesh_topology`
    pytest mark, NOT the test's mesh shape. For example, a (2,2) test uses `topo_marker=
    "mesh-4x2"` because (2,2) and (4,2) both run on the LoudBox "mesh-4x2"-class machine.
    """
<<<<<<< HEAD
    profile = {
        ttnn.FabricConfig.FABRIC_2D: fabric2d_device_params,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y: torus_y_device_params,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY: torus_xy_device_params,
    }[fabric]
    device_params = profile(fabric_payload_size=payload)
=======
    device_params = {
        "fabric_config": fabric,
        "fabric_router_config": create_fabric_router_config(max_payload_size=payload),
    }
>>>>>>> fac3bf836d1 (switching to leaner test changes)
    if reliability_mode is not None:
        device_params["reliability_mode"] = reliability_mode
    return pytest.param(
        shape,
        device_params,
        nlinks,
<<<<<<< HEAD
=======
        topo,
>>>>>>> fac3bf836d1 (switching to leaner test changes)
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topo_marker),
        id=test_id,
    )


ALL_MESH_CONFIGS = [
<<<<<<< HEAD
    # Existing two-chip rows cannot form a useful ring; migrate them one-for-one to Fabric2D.
    _mesh_param(
        (2, 1),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        "linear",
        "fabric2d-2x1-1link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (2, 1),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        2,
        "linear",
        "fabric2d-2x1-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    # Local policy: one 2x2 QuietBox case, canonical 2x4 LoudBox, and one 4x2 axis diagnostic.
    _mesh_param(
        (2, 2),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        "mesh-4x2",
        "fabric2d-mesh-2x2",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (4, 2),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        "mesh-4x2",
        "fabric2d-mesh-4x2",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (4, 2),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        2,
        "mesh-4x2",
        "fabric2d-mesh-4x2-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (2, 4),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        "mesh-4x2",
        "fabric2d-mesh-2x4",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    # Existing QuietBox SP proxy, with former linear/ring siblings collapsed by FabricConfig.
    _mesh_param(
        (4, 1),
=======
    # 2-chip linear
    _mesh_param(
        (2, 1), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 1, ttnn.Topology.Linear, "linear", "linear-2-1link"
    ),
    _mesh_param(
        (2, 1), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 2, ttnn.Topology.Linear, "linear", "linear-2-2link"
    ),
    # 4-chip linear
    _mesh_param(
        (4, 1), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 1, ttnn.Topology.Linear, "linear", "linear-4-1link"
    ),
    _mesh_param(
        (4, 1), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 2, ttnn.Topology.Linear, "linear", "linear-4-2link"
    ),
    # 4-chip ring
    _mesh_param(
        (4, 1), ttnn.FabricConfig.FABRIC_1D_RING, get_max_payload_size(), 1, ttnn.Topology.Ring, "ring", "ring-4-1link"
    ),
    _mesh_param(
        (4, 1), ttnn.FabricConfig.FABRIC_1D_RING, get_max_payload_size(), 2, ttnn.Topology.Ring, "ring", "ring-4-2link"
    ),
    # 2D mesh topologies
    _mesh_param(
        (2, 2), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 1, ttnn.Topology.Linear, "mesh-4x2", "mesh-2x2"
    ),
    _mesh_param(
        (4, 2), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 1, ttnn.Topology.Linear, "mesh-4x2", "mesh-4x2"
    ),
    _mesh_param(
        (2, 4), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 1, ttnn.Topology.Linear, "mesh-4x2", "mesh-2x4"
    ),
    # 8-chip linear
    _mesh_param(
        (8, 1), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 1, ttnn.Topology.Linear, "linear", "linear-8-1link"
    ),
    _mesh_param(
        (8, 1), ttnn.FabricConfig.FABRIC_1D, get_max_payload_size(), 2, ttnn.Topology.Linear, "linear", "linear-8-2link"
    ),
    # 8-chip ring
    _mesh_param(
        (8, 1), ttnn.FabricConfig.FABRIC_1D_RING, get_max_payload_size(), 1, ttnn.Topology.Ring, "ring", "ring-8-1link"
    ),
    _mesh_param(
        (8, 1), ttnn.FabricConfig.FABRIC_1D_RING, get_max_payload_size(), 2, ttnn.Topology.Ring, "ring", "ring-8-2link"
    ),
    # FABRIC_2D variants — every shape with rows>1 AND cols>1 is 2D-eligible.
    # RELAXED_INIT matches the canonical pattern in test_prefill_block.py and is required
    # on BH Galaxy for FABRIC_2D bring-up.
    _mesh_param(
        (2, 2),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        ttnn.Topology.Linear,
        "mesh-4x2",
        "fabric2d-mesh-2x2",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (4, 2),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        ttnn.Topology.Linear,
        "mesh-4x2",
        "fabric2d-mesh-4x2",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (4, 2),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        2,
        ttnn.Topology.Linear,
        "mesh-4x2",
        "fabric2d-mesh-4x2-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (2, 4),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        ttnn.Topology.Linear,
        "mesh-4x2",
        "fabric2d-mesh-2x4",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (8, 4),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        1,
        ttnn.Topology.Linear,
        "mesh-8x4",
        "fabric2d-mesh-8x4-1link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (8, 4),
        ttnn.FabricConfig.FABRIC_2D,
        get_max_payload_size(),
        2,
        ttnn.Topology.Linear,
        "mesh-8x4",
        "fabric2d-mesh-8x4-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    # 8x4 single-galaxy column ring: FABRIC_2D_TORUS_Y wraps the row axis (mesh dim 0, the
    # 8-long SP axis) into a ring; the column axis (dim 1, 4-wide) stays a line. The SP-axis
    # collectives (cluster_axis=0) run Topology.Ring; get_usable_topology keeps Ring because
    # the coords span the full 8-axis (WRAP). FABRIC_2D_TORUS_Y auto-selects
    # single_bh_galaxy_torus_y_graph_descriptor.textproto, whose uniform `channels count: 2`
    # gives the RING-closing row-7<->row-0 edge the same 2-link width as the LINE edges
    # (matches the 110-c78 wiring, where that wrap is a normal 2-link on every column).
    _mesh_param(
        (8, 4),
>>>>>>> fac3bf836d1 (switching to leaner test changes)
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        get_max_payload_size(),
        1,
<<<<<<< HEAD
        "ring",
        "fabric2d-torus-y-4x1-1link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (4, 1),
=======
        ttnn.Topology.Ring,
        "mesh-8x4",
        "fabric2d-torus-y-8x4-1link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (8, 4),
>>>>>>> fac3bf836d1 (switching to leaner test changes)
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        get_max_payload_size(),
        2,
<<<<<<< HEAD
        "ring",
        "fabric2d-torus-y-4x1-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    # Existing LoudBox SP proxy, migrated from the former linear/ring Fabric1d siblings.
    _mesh_param(
        (8, 1),
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        get_max_payload_size(),
        1,
        "ring",
        "fabric2d-torus-y-8x1-1link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (8, 1),
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        get_max_payload_size(),
        2,
        "ring",
        "fabric2d-torus-y-8x1-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    # Galaxy production policy: existing full 8x4 TorusXY, Ring on the exercised SP axis.
=======
        ttnn.Topology.Ring,
        "mesh-8x4",
        "fabric2d-torus-y-8x4-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    # Experiment: TORUS_XY wraps both axes (different deadlock-dateline placement than
    # TORUS_Y). The dispatch op still rings on cluster_axis=0 (the 8-dim). Probing whether
    # the dateline axis affects the multi-hop-over-wrap dispatch hang.
>>>>>>> fac3bf836d1 (switching to leaner test changes)
    _mesh_param(
        (8, 4),
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        get_max_payload_size(),
<<<<<<< HEAD
        2,
        "mesh-8x4",
        "fabric2d-torus-xy-8x4-2link",
=======
        1,
        ttnn.Topology.Ring,
        "mesh-8x4",
        "fabric2d-torus-xy-8x4-1link",
>>>>>>> fac3bf836d1 (switching to leaner test changes)
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (8, 4),
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        get_max_payload_size(),
        2,
        ttnn.Topology.Ring,
        "mesh-8x4",
        "fabric2d-torus-xy-8x4-2link",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
    _mesh_param(
        (8, 4),
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        get_max_payload_size(64),
        2,
        ttnn.Topology.Ring,
        "mesh-8x4",
        "fabric2d-torus-xy-8x4-2link-expanded_fabric_payload",
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
    ),
]


def fabric_to_device_params(fabric_cfg):
    assert fabric_cfg in (
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    )
    if fabric_cfg == ttnn.FabricConfig.FABRIC_1D:
        return fabric_1d_device_params()
    if fabric_cfg == ttnn.FabricConfig.FABRIC_2D:
        return fabric2d_device_params()
    if fabric_cfg == ttnn.FabricConfig.FABRIC_2D_TORUS_X:
        return torus_x_device_params()
    if fabric_cfg == ttnn.FabricConfig.FABRIC_2D_TORUS_Y:
        return torus_y_device_params()
    return torus_xy_device_params()
