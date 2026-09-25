# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fabric topology selection for the 8x4 Blackhole Galaxy, from one env knob.

``MISTRAL_FABRIC=linear`` (default): plain mesh graph descriptor + ``FABRIC_1D`` + ``Topology.Linear``,
which maps on any galaxy, torus-wired or not. ``MISTRAL_FABRIC=ring``: torus-xy descriptor +
``FABRIC_1D_RING`` + ``Topology.Ring`` for the CCLs (a perf lever only; never a correctness gate).
The mesh graph descriptor must be exported before the cluster initialises, which is why
``apply_mesh_graph_descriptor`` runs from the package conftest at import time.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTORS = {
    "linear": "single_bh_galaxy_mesh_graph_descriptor.textproto",
    "ring": "single_bh_galaxy_torus_xy_graph_descriptor.textproto",
}


def fabric_mode() -> str:
    mode = os.environ.get("MISTRAL_FABRIC", "linear").strip().lower()
    if mode not in _DESCRIPTORS:
        raise ValueError(f"MISTRAL_FABRIC must be one of {sorted(_DESCRIPTORS)}, got {mode!r}")
    return mode


def mesh_graph_descriptor_path(mode=None) -> str:
    root = Path(os.environ.get("TT_METAL_HOME", _REPO_ROOT))
    return str(root / "tt_metal" / "fabric" / "mesh_graph_descriptors" / _DESCRIPTORS[mode or fabric_mode()])


def apply_mesh_graph_descriptor() -> str:
    """Export TT_MESH_GRAPH_DESC_PATH for the selected topology unless the caller already set one."""
    os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", mesh_graph_descriptor_path())
    return os.environ["TT_MESH_GRAPH_DESC_PATH"]


def fabric_config():
    import ttnn

    return ttnn.FabricConfig.FABRIC_1D_RING if fabric_mode() == "ring" else ttnn.FabricConfig.FABRIC_1D


def ccl_topology():
    import ttnn

    return ttnn.Topology.Ring if fabric_mode() == "ring" else ttnn.Topology.Linear
