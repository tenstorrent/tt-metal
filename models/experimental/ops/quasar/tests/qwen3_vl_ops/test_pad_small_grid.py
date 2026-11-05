# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_pad.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call pads a [2766, 2560] activation (not a whole number of tile rows) up to
# 4096 rows. This keeps the unaligned row count by padding [70, 512] up to 128 rows.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.pad`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.pad
_NAME = "ttnn.pad"

CASES = [
    case("small_70_to_128x512", _NAME, [t([70, 512]), lit([[0, 58], [0, 0]]), lit(0)], {}, [out([128, 512])]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_pad_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
