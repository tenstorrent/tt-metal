# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_unsqueeze.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured calls unsqueeze [4096, 2560] prefill activations at dims 0 and 1. These keep
# both dims at [128, 512].
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.unsqueeze`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.unsqueeze
_NAME = "ttnn.unsqueeze"

CASES = [
    case("small_dim1_1x128x512", _NAME, [t([1, 128, 512]), lit(1)], {}, [out([1, 1, 128, 512])]),
    case("small_dim0_128x512", _NAME, [t([128, 512]), lit(0)], {}, [out([1, 128, 512])]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_unsqueeze_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
