# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 pins: DiT joint / cross variants on the 4-chip ring.

- sharded joint (R5): joint K/V slices travel the ring as a second transport entry kind; a partial last slice
  (logical_l % 32 != 0) and a slice whose valid rows end mid-tile exercise the joint key-tail mask tile;
- non-aligned replicated joint (L % 32 != 0): second key-tail mask tile, joint q-block tail;
- cross-attention (n_q != n_kv): Q tile count decoupled from the K/V tile count.
Both DEST widths (the fp32 helper body and the streamed bf16 body apply the mask differently).
"""

from __future__ import annotations

import pytest
import ttnn

from eval.golden_tests.ring_joint_scaled_dot_product_attention.helpers import (
    run_ring_joint_scaled_dot_product_attention,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import ring_mock_meshes


def _ring():
    for mesh_shape, axis, configs in ring_mock_meshes():
        if axis == 1 and ttnn.FabricConfig.FABRIC_2D_TORUS_XY in configs and mesh_shape[1] >= 3:
            return [(mesh_shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY)]
    return []


# (id, inputs (Q, K, V[, LOGICAL joint Q]), joint_mode)
CASES = [
    ("sharded_L333", ((1, 6, 512, 128), (1, 6, 512, 128), (1, 6, 512, 128), (1, 6, 333, 128)), "sharded"),
    ("sharded_L512", ((1, 4, 256, 64), (1, 4, 256, 64), (1, 4, 256, 64), (1, 4, 512, 64)), "sharded"),
    ("sharded_L118", ((1, 6, 1024, 128), (1, 6, 1024, 128), (1, 6, 1024, 128), (1, 6, 118, 128)), "sharded"),
    ("replicated_L333", ((1, 10, 512, 64), (1, 10, 512, 64), (1, 10, 512, 64), (1, 10, 333, 64)), "replicated"),
    ("replicated_L118", ((1, 6, 1024, 128), (1, 6, 1024, 128), (1, 6, 1024, 128), (1, 6, 118, 128)), "replicated"),
    ("cross_q32", ((1, 8, 32, 64), (1, 8, 256, 64), (1, 8, 256, 64)), "none"),
    ("cross_q64", ((1, 16, 64, 64), (1, 16, 512, 64), (1, 16, 512, 64)), "none"),
]


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("case_id,inputs,joint_mode", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("mesh_shape,fabric_config", _ring())
def test_joint_cross(mesh_shape, fabric_config, case_id, inputs, joint_mode, fp32_dest_acc_en, mesh_device):
    assert tuple(mesh_device.shape) == tuple(mesh_shape)
    run_ring_joint_scaled_dot_product_attention(
        inputs,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        fp32_dest_acc_en=fp32_dest_acc_en,
        layout=ttnn.TILE_LAYOUT,
        joint_mode=joint_mode,
        mask_mode="none",
        is_balanced=False,
        seq_padding="padded",
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        extras={"num_links": 2},
    )
