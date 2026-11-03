# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_untilize.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call untilizes a [1,1,32,37984] bfloat8_b logits slice. This keeps the
# multicore untilize in bfloat16 at widths below and above 64 tiles (2048 and 2560).
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.untilize`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.untilize
_NAME = "ttnn.untilize"

CASES = [
    case(
        "small_32x2048",
        _NAME,
        [t([1, 1, 32, 2048])],
        {"use_multicore": lit(True), "sub_core_grids": lit(None)},
        [out([1, 1, 32, 2048], layout="ROW_MAJOR")],
    ),
    case(
        "small_32x2560",
        _NAME,
        [t([1, 1, 32, 2560])],
        {"use_multicore": lit(True), "sub_core_grids": lit(None)},
        [out([1, 1, 32, 2560], layout="ROW_MAJOR")],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_untilize_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
