# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fabric / collective-topology selection for the M3 harnesses, from environment variables.

Shared by ``tests/galaxy_prefill_kv_pcc.py`` and ``tests/perf/profile_prefill.py`` so both spell the knobs
the same way as the production runner (``PREFILL_FABRIC_MODE`` in
``models/demos/common/prefill/runners/runner_utils.py``) and reject a typo with a readable message.
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
