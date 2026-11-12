# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_slice.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call slices one tile row ([2752, 2784)) out of a [1,1,4096,2560] prefill
# activation. This takes a tile row from the middle of a [1,1,128,512] tensor.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.slice`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.slice
_NAME = "ttnn.slice"

CASES = [
    case(
        "small_rows_64_96_of_128x512",
        _NAME,
        [t([1, 1, 128, 512]), lit([0, 0, 64, 0]), lit([1, 1, 96, 512])],
        {},
        [out([1, 1, 32, 512])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_slice_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
