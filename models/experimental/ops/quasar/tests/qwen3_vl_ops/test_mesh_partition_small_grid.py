# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_mesh_partition.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call partitions a [2752, 2560] activation along dim 1 across the mesh. This
# keeps that call at [128, 512] (on a 1x1 mesh the partition is the whole tensor).
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.mesh_partition`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.mesh_partition
_NAME = "ttnn.mesh_partition"

CASES = [
    case("small_dim1_128x512", _NAME, [t([128, 512]), lit(1)], {}, [out([128, 512])]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_mesh_partition_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
