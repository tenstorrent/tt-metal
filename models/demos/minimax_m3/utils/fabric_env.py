# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fabric / collective-topology selection for the M3 harnesses, from environment variables.

Shared by ``tests/galaxy_prefill_kv_pcc.py`` and ``tests/perf/profile_prefill.py`` so both spell the knobs
the same way as the production runner (``PREFILL_FABRIC_MODE`` in
``models/demos/common/prefill/runners/runner_utils.py``) and reject a typo with a readable message.

The ``M3_MOE_*`` knobs select the routed-expert MoE's EP transport (tt/mlp.py -> TtMiniMaxMoE); both
default to today's path.
"""

import os

import ttnn

FABRIC_CONFIGS = {
    "1d": ttnn.FabricConfig.FABRIC_1D,
    "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "2d_torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
}

CCL_TOPOLOGIES = {
    "linear": ttnn.Topology.Linear,
    "ring": ttnn.Topology.Ring,
}

MOE_COMBINE_VERSIONS = {"v1": "v1", "v2": "v2"}

# combine_fabric2d walks the ring on cluster_axis=0 over real wrap links, so it needs a 2D fabric that
# wraps the SP (row) axis.
MOE_COMBINE_V2_FABRICS = (ttnn.FabricConfig.FABRIC_2D_TORUS_Y, ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
# Fabrics that wrap axis 0, so a Ring on cluster_axis=0 has its wrap link.
AXIS0_RING_FABRICS = (ttnn.FabricConfig.FABRIC_1D_RING,) + MOE_COMBINE_V2_FABRICS


def _lookup(table, var, default):
    value = os.getenv(var, "").strip().lower() or default
    if value not in table:
        raise ValueError(f"{var} must be one of {sorted(table)} (got {os.getenv(var)!r})")
    return table[value]


def fabric_config_from_env(var="M3_FABRIC", default="1d"):
    """``M3_FABRIC=1d|1d_ring|2d|2d_torus_xy`` -> ttnn.FabricConfig (unset / empty -> ``default``)."""
    return _lookup(FABRIC_CONFIGS, var, default)


def ccl_topology_from_env(var="M3_CCL_TOPOLOGY", default="linear"):
    """``M3_CCL_TOPOLOGY=linear|ring`` (case-insensitive) -> ttnn.Topology for the legacy CCLs.

    ``high_bw_all_gather`` ignores it and derives ring vs line from the fabric; Ring needs a ring / torus
    fabric or the legacy CCLs hang.
    """
    return _lookup(CCL_TOPOLOGIES, var, default)


def moe_topology_from_env(var="M3_MOE_TOPOLOGY", default="linear"):
    """``M3_MOE_TOPOLOGY=linear|ring`` -> ttnn.Topology for the MoE's axis-0 EP dispatch and v1 combine.

    Separate from ``M3_CCL_TOPOLOGY``: the TP (axis-1) collectives around the MoE keep their own knob.
    Ring needs a fabric that wraps axis 0 (``M3_FABRIC=1d_ring`` or ``2d_torus_xy``).
    """
    return _lookup(CCL_TOPOLOGIES, var, default)


def moe_combine_from_env(var="M3_MOE_COMBINE", default="v1"):
    """``M3_MOE_COMBINE=v1|v2`` -> ``"v1"`` (deepseek_prefill.combine) or ``"v2"`` (combine_fabric2d).

    v2 needs ``M3_FABRIC=2d_torus_xy`` and a raised fabric max payload; use set_fabric_config_from_env().
    """
    return _lookup(MOE_COMBINE_VERSIONS, var, default)


def fabric_router_config_from_env():
    """Router config the fabric must open with, or None to keep the fabric default.

    Only ``M3_MOE_COMBINE=v2`` needs one: combine_fabric2d sends a whole bf16 token plus a 64 B routing
    tail in one packet (6144 * 2 + 64 = 12352 B for M3), above the 4352 B default. It takes DeepSeek's
    payload size (the value combine_fabric2d is tested with).
    """
    if moe_combine_from_env() != "v2":
        return None
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size

    return create_fabric_router_config(max_payload_size=get_max_payload_size())


def set_fabric_config_from_env(fabric_config=None):
    """``ttnn.set_fabric_config`` for ``M3_FABRIC`` (or ``fabric_config``), plus the router config v2 needs.

    Without ``M3_MOE_COMBINE=v2`` this is exactly ``ttnn.set_fabric_config(fabric_config)``. Rejects a
    fabric the MoE knobs cannot run on before the device opens, rather than hanging or failing mid-load.
    """
    if fabric_config is None:
        fabric_config = fabric_config_from_env()
    if moe_topology_from_env() == ttnn.Topology.Ring and fabric_config not in AXIS0_RING_FABRICS:
        raise ValueError(
            f"M3_MOE_TOPOLOGY=ring needs a fabric that wraps axis 0 ({[str(f) for f in AXIS0_RING_FABRICS]}), "
            f"got {fabric_config}"
        )
    router_config = fabric_router_config_from_env()
    if router_config is None:
        ttnn.set_fabric_config(fabric_config)
        return
    if fabric_config not in MOE_COMBINE_V2_FABRICS:
        raise ValueError(
            f"M3_MOE_COMBINE=v2 needs a fabric that wraps the SP axis ({[str(f) for f in MOE_COMBINE_V2_FABRICS]}), "
            f"got {fabric_config}; set M3_FABRIC=2d_torus_xy"
        )
    ttnn.set_fabric_config(fabric_config, router_config=router_config)
