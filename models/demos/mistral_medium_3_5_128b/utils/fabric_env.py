# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fabric / collective-topology selection from the environment. Ported from
``minimax_m3/utils/fabric_env.py`` (same 8x4 bh_galaxy target).

The recipe's rule: default to the plain mesh descriptor with ``FABRIC_1D`` + ``Topology.Linear``,
which maps on **any** galaxy, torus-wired or not. The torus is the one significant perf lever
bring-up has, so it sits behind a single env knob — but it is never a correctness gate, and a
bring-up must not fail for the want of it. If a torus descriptor will not map, log it as ``env``,
fall back to linear, and carry on.

    MISTRAL_FABRIC=1d|1d_ring|2d|2d_torus_xy   (default: 1d)
    MISTRAL_CCL_TOPOLOGY=linear|ring           (default: linear)

``Topology.Ring`` needs a ring / torus fabric or the legacy CCLs hang on an unwrapped axis, so the
two knobs must be moved together.
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


def fabric_config_from_env(var="MISTRAL_FABRIC", default="1d"):
    """``MISTRAL_FABRIC=1d|1d_ring|2d|2d_torus_xy`` -> ttnn.FabricConfig (unset / empty -> default)."""
    return _lookup(FABRIC_CONFIGS, var, default)


def ccl_topology_from_env(var="MISTRAL_CCL_TOPOLOGY", default="linear"):
    """``MISTRAL_CCL_TOPOLOGY=linear|ring`` (case-insensitive) -> ttnn.Topology for the CCLs."""
    return _lookup(CCL_TOPOLOGIES, var, default)


def topology_name():
    """Short label for the topology a measurement was taken on — the README records this per
    measurement, because linear and torus collective costs are not comparable."""
    return f"{os.getenv('MISTRAL_FABRIC', '1d').strip().lower() or '1d'}/{os.getenv('MISTRAL_CCL_TOPOLOGY', 'linear').strip().lower() or 'linear'}"
