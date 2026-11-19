# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_scatter.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call scatters a [2752, 2560] bfloat16 source along dim 0 into a [2766, 2560]
# tensor using an INT32 row-major index. This keeps that form at [64, 512] into [70, 512].
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.scatter`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.scatter
_NAME = "ttnn.scatter"

CASES = [
    case(
        "small_dim0_64_into_70x512",
        _NAME,
        [t([70, 512]), lit(0), t([64, 512], dtype="INT32", layout="ROW_MAJOR"), t([64, 512])],
        {},
        [out([70, 512])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_scatter_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
