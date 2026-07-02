# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Ring-routing probe for all_gather (Refinement 3). DO NOT DELETE.

The WH sim topology `wh_t3k_allmmio_all_gather` uses the RING-typed t3k_1x8
mesh-graph descriptor. This probe asks `ccl_dm_route` what route it computes for
Topology.Ring vs Linear on adjacent and WRAPAROUND (7<->0) device pairs — the
addressing prerequisite for a single-direction ring all-gather. It performs NO
fabric transfer (pure route query), so it can't hang on data-plane issues; it
only tells us whether the ring wraparound resolves to a 1-hop neighbour.
"""

import pytest
import ttnn
from loguru import logger


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_ring_routes(mesh_device):
    N = mesh_device.shape[1]
    pairs = [(0, 1), (3, 4), (6, 7), (7, 0), (0, 7), (0, N - 1)]
    for topo in (ttnn.Topology.Linear, ttnn.Topology.Ring):
        for a, b in pairs:
            if b >= N:
                continue
            try:
                r = ttnn._ttnn.fabric.ccl_dm_route(
                    mesh_device, ttnn.MeshCoordinate(0, a), ttnn.MeshCoordinate(0, b), topo
                )
                logger.info(
                    f"[route] {topo} {a}->{b}: num_hops={r.num_hops} is_forward={r.is_forward} neighbor_id={r.neighbor_id}"
                )
            except Exception as e:  # noqa: BLE001
                logger.info(f"[route] {topo} {a}->{b}: EXCEPTION {type(e).__name__}: {e}")
