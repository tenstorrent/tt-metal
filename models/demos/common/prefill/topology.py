# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fabric-to-collective-topology mapping for prefill code."""

from typing import Optional

import ttnn

# Mesh dim 0 = rows = Y = SP; dim 1 = columns = X = TP.
_FABRIC_PER_AXIS_TOPOLOGY = {
    ttnn.FabricConfig.FABRIC_2D_TORUS_X: (ttnn.Topology.Linear, ttnn.Topology.Ring),
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y: (ttnn.Topology.Ring, ttnn.Topology.Linear),
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY: (ttnn.Topology.Ring, ttnn.Topology.Ring),
    # Compatibility only; scoped prefill production and tests reject Fabric1D.
    ttnn.FabricConfig.FABRIC_1D_RING: (ttnn.Topology.Ring, ttnn.Topology.Linear),
}


def per_axis_topology(
    fabric_config: Optional[ttnn.FabricConfig] = None,
) -> tuple[ttnn.Topology, ttnn.Topology]:
    """Derive ``(SP, TP)`` collective topology from the requested or active fabric."""
    if fabric_config is None:
        fabric_config = ttnn.get_fabric_config()
    mapped = _FABRIC_PER_AXIS_TOPOLOGY.get(fabric_config)
    if mapped is not None:
        return mapped
    name = getattr(fabric_config, "name", str(fabric_config))
    if "TORUS" in name.upper() or "RING" in name.upper():
        raise ValueError(f"per_axis_topology: wrap-capable fabric {name} has no explicit per-axis topology mapping")
    return (ttnn.Topology.Linear, ttnn.Topology.Linear)


def assert_torus_xy_topology(
    fabric_config: Optional[ttnn.FabricConfig] = None,
) -> tuple[ttnn.Topology, ttnn.Topology]:
    """Require production TorusXY's Ring/Ring topology and return it."""
    topology = per_axis_topology(fabric_config)
    expected = (ttnn.Topology.Ring, ttnn.Topology.Ring)
    if topology != expected:
        active = fabric_config if fabric_config is not None else ttnn.get_fabric_config()
        raise ValueError(f"production prefill requires TorusXY Ring/Ring, got fabric={active}, topology={topology}")
    return topology
