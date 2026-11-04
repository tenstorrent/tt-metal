# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 pins: causal prefill (contiguous and zig-zag balanced) with shared-K/V heads on the 4-chip ring.

Longer sequences than the golden matrix so every unit walks several key blocks per source: the causal key plan
(per-unit truncation of each source, sources wholly in the future skipped, diagonal / dead mask tiles inside the kept
blocks) and the stacked-head units (q_heads_per_unit = hq / hk heads sharing one key-block stream) are exercised
across block boundaries, on both DEST widths. Also the profiling target for the causal busiest-chip measurement.
"""

from __future__ import annotations

import pytest
import ttnn

# mesh_device / _fabric_mesh_cache come from this directory's conftest.py (one session-wide fabric-mesh cache).
from eval.golden_tests.ring_joint_scaled_dot_product_attention.helpers import (
    run_ring_joint_scaled_dot_product_attention,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import ring_mock_meshes


def _ring():
    for mesh_shape, axis, configs in ring_mock_meshes():
        if axis == 1 and ttnn.FabricConfig.FABRIC_2D_TORUS_XY in configs and mesh_shape[1] >= 3:
            return [(mesh_shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY)]
    return []


# (id, per-device Q / K / V shapes)
SHAPES = [
    ("gqa8_n2048", ((1, 8, 2048, 128), (1, 1, 2048, 128), (1, 1, 2048, 128))),
    ("gqa2_n1024", ((1, 8, 1024, 128), (1, 4, 1024, 128), (1, 4, 1024, 128))),
    ("mla_n1024", ((1, 8, 1024, 320), (1, 1, 1024, 320), (1, 8, 1024, 128))),
]


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("is_balanced", [False, True], ids=["contiguous", "balanced"])
@pytest.mark.parametrize("shape_id,inputs", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("mesh_shape,fabric_config", _ring())
def test_causal_heads(mesh_shape, fabric_config, shape_id, inputs, is_balanced, fp32_dest_acc_en, mesh_device):
    assert tuple(mesh_device.shape) == tuple(mesh_shape)
    run_ring_joint_scaled_dot_product_attention(
        inputs,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat8_b,
        fp32_dest_acc_en=fp32_dest_acc_en,
        layout=ttnn.TILE_LAYOUT,
        joint_mode="none",
        mask_mode="causal",
        is_balanced=is_balanced,
        seq_padding="padded",
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        extras={"num_links": 2, "sample_rows": 128},
    )
