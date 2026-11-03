# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression: consecutive calls whose port placement differs (topology / num_links / shape) must stay correct.

Verifier finding (Phase 0): the compute-core reader NoC-read a port's arrival counter into its OWN copy of the
global arrival semaphore. Port cores are re-placed per (topology, num_links), so a core that polled as a compute
core in one call could be a port core in the next one and start with a stale arrival count — its relay and its
consumers then read landing slots before the data landed (golden ring-mock: Linear 2-link calls followed by a Ring
1-link joint call, PCC 0.71-0.84 or garbage). The poll now reads into a dedicated scratch semaphore. This test
replays the golden ring-mock order on the 4-chip snake ring, every step checked against the reference.
"""

from __future__ import annotations

import pytest
import ttnn

from eval.feature_matrix import apply_input_taggers
from eval.golden_tests.ring_joint_scaled_dot_product_attention.conftest import (  # noqa: F401 — fixtures
    _fabric_mesh_cache,
    mesh_device,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.helpers import (
    run_ring_joint_scaled_dot_product_attention,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import ring_mock_meshes
from ttnn.operations.ring_joint_scaled_dot_product_attention import INPUT_TAGGERS

_BASE = {
    "dtype": ttnn.bfloat16,
    "kv_dtype": ttnn.bfloat16,
    "fp32_dest_acc_en": True,
    "layout": ttnn.TILE_LAYOUT,
    "mask_mode": "none",
    "is_balanced": False,
    "seq_padding": "padded",
}
_NOJOINT = ((1, 10, 512, 128), (1, 10, 512, 128), (1, 10, 512, 128))
_JOINT = ((1, 6, 512, 128), (1, 6, 512, 128), (1, 6, 512, 128), (1, 6, 512, 128))
# the golden ring-mock order that exposed the race: every no-joint placement, then every joint placement
_SEQUENCE = [
    (inp, topo, links)
    for inp in (_NOJOINT, _JOINT)
    for topo in (ttnn.Topology.Ring, ttnn.Topology.Linear)
    for links in (1, 2)
]
_RING_1XN = [
    (shape, axis)
    for shape, axis, cfgs in ring_mock_meshes()
    if axis == 1 and ttnn.FabricConfig.FABRIC_2D_TORUS_XY in cfgs
]


@pytest.mark.skipif(not _RING_1XN, reason="no 1xN snake ring on this cluster")
@pytest.mark.parametrize("step", range(len(_SEQUENCE)))
@pytest.mark.parametrize(
    "mesh_shape,fabric_config,cluster_axis",
    [(shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY, axis) for shape, axis in _RING_1XN[:1]],
)
def test_placement_switch(mesh_shape, fabric_config, cluster_axis, step, mesh_device):
    inputs, topology, num_links = _SEQUENCE[step]
    axes = {
        **_BASE,
        "joint_mode": "replicated" if len(inputs) == 4 else "none",
        "cluster_axis": cluster_axis,
        "topology": topology,
    }
    axes.update(apply_input_taggers(INPUT_TAGGERS, inputs, axes))
    run_ring_joint_scaled_dot_product_attention(
        inputs, mesh_device=mesh_device, extras={"num_links": num_links}, **axes
    )
