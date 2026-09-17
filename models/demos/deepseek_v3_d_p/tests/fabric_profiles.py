# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared device-fixture profiles for scoped prefill tests.

Return a fresh dictionary for every ``pytest.param`` so cases cannot share mutable fixture state.
"""

import re
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


def fabric2d_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Unwrapped local Fabric2D profile for 2x2/2x4/4x2 Blackhole tests."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def fabric_1d_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Unwrapped 1D fabric profile."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def torus_y_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Fabric2D Ring/Linear profile for an Nx1 mesh."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def torus_x_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Fabric2D Linear/Ring profile for a 1xN mesh."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def torus_xy_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Production 8x4 Ring/Ring profile.

    Leave `TT_MESH_GRAPH_DESC_PATH` unset. A torus descriptor declares its channel counts
    `policy: STRICT`, which prunes any physical pair carrying fewer channels than declared,
    so a box one degraded link down fails the whole map; auto-discovery validates RELAXED
    and maps the wrap. What auto-discovery gives up is an error on a wrap it cannot cable,
    substituting a lesser fabric instead, which is what `assert_requested_tp_wrap_was_realized`
    holds it to.
    """
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def assert_torus_xy_descriptor(path: str) -> None:
    """Fail before device open unless every declared device mesh is 8x4 Ring/Ring."""
    descriptor_path = Path(path).resolve()
    text = descriptor_path.read_text()
    declared = len(re.findall(r"device_topology\s*\{", text))
    topologies = re.findall(
        r"device_topology\s*\{[^}]*dims:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\][^}]*"
        r"dim_types:\s*\[\s*(\w+)\s*,\s*(\w+)\s*\]",
        text,
    )
    assert topologies, f"{descriptor_path}: no two-dimensional device topology found"
    assert len(topologies) == declared, (
        f"{descriptor_path}: {declared} device_topology block(s) declared but only "
        f"{len(topologies)} are two-dimensional; TorusXY requires every mesh to be 8x4 Ring/Ring"
    )
    assert all(
        topology == ("8", "4", "RING", "RING") for topology in topologies
    ), f"{descriptor_path}: TorusXY requires every device topology to be 8x4 Ring/Ring; found {topologies}"


_REQUESTS_TP_WRAP = frozenset(
    {
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    }
)


def tp_axis_is_wrapped(mesh_device) -> bool:
    """Whether the fabric that came up rings the TP axis, read from the routes it resolved.

    The mesh graph descriptor cannot answer this: it is absent whenever the graph came from
    auto-discovery, which is the one path that substitutes a fabric for the requested one.
    Routing tables are always populated, and they are what the collective actually consumes,
    so a wrap is detected by where a route goes rather than by what the topology claims. The
    far peer sits one hop behind on a ring and `cols - 1` hops ahead on a line.
    """
    _, cols = tuple(mesh_device.shape)
    mesh_shapes = ttnn.get_physical_mesh_shapes()
    assert len(mesh_shapes) == 1, f"expected a single local mesh, got {mesh_shapes}"
    mesh_id = ttnn.MeshId(next(iter(mesh_shapes)))

    def direction(src, dst):
        node = lambda col: ttnn.FabricNodeId(mesh_id, col)  # noqa: E731 -- row 0, so chip id is the column
        value = ttnn.get_eth_forwarding_direction(node(src), node(dst))
        assert value is not None, f"no TP route from column {src} to {dst}"
        return value

    return direction(0, cols - 1) == direction(1, 0)


def assert_requested_tp_wrap_was_realized(mesh_device) -> None:
    """Fail an arm whose wrap the control plane dropped on its way to the device.

    Auto-discovery matches the requested fabric against the cabling it finds and falls back
    TORUS_XY -> TORUS_Y -> TORUS_X -> MESH behind a log warning rather than raising. A box
    short one wrap link therefore answers a torus request with an unwrapped fabric and runs
    the arm to green on the single topology that arm exists to exercise.
    """
    if ttnn.get_fabric_config() not in _REQUESTS_TP_WRAP:
        return

    _, cols = tuple(mesh_device.shape)
    if cols <= 2:
        # `is_genuine_torus_dim`: a dimension this narrow keeps ordinary mesh links, so the
        # fabric is right to leave it unwrapped and there is nothing here to hold it to.
        return

    assert tp_axis_is_wrapped(mesh_device), (
        f"{ttnn.get_fabric_config()} was requested but the TP axis came up unwrapped; "
        f"this box cannot gate a wrap-direction bug -- check the wrap cabling"
    )
