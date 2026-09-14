# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package fixtures — and, critically, the mesh-graph descriptor choice.

**The descriptor must be chosen before the cluster initialises**, which is why it is set at import
time here and not inside a fixture: a torus descriptor cannot map on a pod without wrap-around links,
and the failure happens during control-plane discovery, long before any test body runs.

Default: the plain 8x4 mesh descriptor with ``FABRIC_1D`` + ``ttnn.Topology.Linear``, which maps on
**any** galaxy, torus-wired or not. ``LLAMA_TORUS=1`` selects the torus descriptor
(``FABRIC_1D_RING`` + ``Topology.Ring``) — the one significant perf lever bring-up has, taken where
the pod offers it, never a correctness gate. If the torus descriptor will not map, that is an ``env``
log line and a fall back to linear, not a failure.

Topology is recorded with every measurement in ``README.md``: linear and torus collective costs are
not comparable, so an unlabelled number is a misleading number.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[4]
DESCRIPTORS = REPO_ROOT / "tt_metal" / "fabric" / "mesh_graph_descriptors"
LINEAR_DESC = DESCRIPTORS / "single_bh_galaxy_mesh_graph_descriptor.textproto"
TORUS_DESC = DESCRIPTORS / "single_bh_galaxy_torus_x_graph_descriptor.textproto"


def use_torus() -> bool:
    return os.getenv("LLAMA_TORUS", "0").strip().lower() in ("1", "true", "yes", "on")


def _select_descriptor() -> None:
    """Point the control plane at our descriptor unless the caller already pinned one."""
    if os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        return
    desc = TORUS_DESC if use_torus() else LINEAR_DESC
    if desc.exists():
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(desc)


_select_descriptor()

# Fabric + CCL topology follow the descriptor; the CCL manager and every collective read this pair.
FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D_RING if use_torus() else ttnn.FabricConfig.FABRIC_1D
CCL_TOPOLOGY = ttnn.Topology.Ring if use_torus() else ttnn.Topology.Linear
TOPOLOGY_NAME = "torus" if use_torus() else "linear"


@pytest.fixture(scope="session")
def topology_name() -> str:
    """Which fabric wiring this run measured on — quoted next to every number."""
    return TOPOLOGY_NAME


@pytest.fixture(scope="session")
def ccl_topology():
    return CCL_TOPOLOGY


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Every test draws its random weights from a pinned seed, so a PCC number is reproducible and
    the two sides of a comparison are handed the same tensors."""
    torch.manual_seed(1234)
    yield
