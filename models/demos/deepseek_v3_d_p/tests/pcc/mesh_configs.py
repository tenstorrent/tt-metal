# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared mesh configuration parameters for dispatch/combine PCC tests.

Both test_prefill_dispatch.py, test_prefill_combine.py, and test_ttnn_dispatch_combine.py
import ALL_MESH_CONFIGS to avoid duplicating the same pytest.param entries.

Topology vs FabricConfig
------------------------
`Topology` is the CCL algorithm's data-flow shape (Linear / Ring / Mesh / Torus) — a
pytest parametrize axis. `FabricConfig` is the device-level fabric wiring (FABRIC_1D /
FABRIC_1D_RING / FABRIC_2D) — set via the `mesh_device` fixture's device_params. The two
are orthogonal: e.g. `Topology::Linear + FABRIC_2D` is valid and intended — it asks for
linear data-flow over a 2D-routed fabric.

Test-id naming convention
-------------------------
- FABRIC_1D entries use shape-only ids: `linear-N-Llink`, `ring-N-Llink`, `mesh-RxC`.
- FABRIC_2D entries are prefixed with `fabric2d-`: `fabric2d-mesh-RxC[-Llink]`.

CI -k filters depend on this convention. For example, `-k 'mesh-8x4'` matches BOTH 1D and
2D variants because `mesh-8x4` is a substring of `fabric2d-mesh-8x4`. To run only 1D
under an existing `-k 'mesh-*'` filter, append `and not fabric2d-`. To run only 2D, use
a positive `and fabric2d-` filter.
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


def _mesh_param(shape, fabric, payload, nlinks, topo, topo_marker, test_id, reliability_mode=None):
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
        topo,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topo_marker),
        id=test_id,
    )


ALL_MESH_CONFIGS = [
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
]
