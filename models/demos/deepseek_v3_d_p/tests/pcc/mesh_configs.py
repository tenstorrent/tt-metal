# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared mesh configuration parameters for dispatch/combine PCC tests.

The op_unit_tests test_prefill_dispatch.py, test_prefill_combine.py, and
test_ttnn_dispatch_combine.py import ALL_MESH_CONFIGS to avoid duplicating the same
pytest.param entries.

FabricConfig is the single source of truth. Consumers derive their cluster-axis CCL
topology with ``per_axis_topology`` instead of carrying a second parameter.

Test-id naming convention
-------------------------
- Local IDs are `fabric2d-mesh-2x2`, `fabric2d-mesh-2x4`, and the single
  `fabric2d-mesh-4x2-axis` diagnostic.
- The production ID is `fabric2d-torus-xy-8x4-2link` and must be selected exactly.
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


def _mesh_param(shape, fabric, payload, nlinks, topo_marker, test_id, reliability_mode=None):
    """Build a single pytest.param for the mesh_device parametrize axis.

    `topo_marker` is the CI hardware-class string consumed by the `requires_mesh_topology`
    pytest mark, NOT the test's mesh shape. For example, a (2,2) test uses `topo_marker=
    "mesh-4x2"` because (2,2) and (4,2) both run on the LoudBox "mesh-4x2"-class machine.
    """
    device_params = {
        "fabric_config": fabric,
        "fabric_router_config": create_fabric_router_config(max_payload_size=payload),
    }
    if reliability_mode is not None:
        device_params["reliability_mode"] = reliability_mode
    return pytest.param(
        shape,
        device_params,
        nlinks,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topo_marker),
        id=test_id,
    )


ALL_MESH_CONFIGS = [
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
        "fabric2d-mesh-4x2-axis",
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
    # Galaxy production policy: existing full 8x4 TorusXY, Ring on the exercised SP axis.
    _mesh_param(
        (8, 4),
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        get_max_payload_size(),
        2,
        "mesh-8x4",
        "fabric2d-torus-xy-8x4-2link",
        reliability_mode=ttnn.FabricReliabilityMode.STRICT_INIT,
    ),
]
