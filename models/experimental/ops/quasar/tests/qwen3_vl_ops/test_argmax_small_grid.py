# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_argmax.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call is greedy decoding over the full [1,1,32,151936] vocab. argmax compares
# every element in scalar code on a data-movement core, so 4.9M elements do not finish on
# craq-sim in the test budget. These keep the full vocab width for two rows (the reduction and
# its cross-core merge are the same) and add a 32-row case at a narrower width. At least two
# output rows are needed: with one, argmax_multi_core cannot split rows across both DM
# processors and binds its DFBs self-looped on one, which Quasar rejects (dfb.cpp:603).
# The model always has 32 rows (one per decode user), so it never takes that path.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.argmax`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, lit

_OP = ttnn.argmax
_NAME = "ttnn.argmax"

CASES = [
    case(
        "small_2x151936",
        _NAME,
        [t([1, 1, 2, 151936], layout="ROW_MAJOR")],
        {"dim": lit(-1), "output_tensor": lit(None), "keepdim": lit(False), "sub_core_grids": lit(None)},
    ),
    case(
        "small_32x4096",
        _NAME,
        [t([1, 1, 32, 4096], layout="ROW_MAJOR")],
        {"dim": lit(-1), "output_tensor": lit(None), "keepdim": lit(False), "sub_core_grids": lit(None)},
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_argmax_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
