# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_split.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call splits the [1,1,32,151936] bfloat8_b LM-head logits from L1 into four
# pieces along the width. This keeps that form in bfloat16 at [1,1,32,4096]. With no
# memory_config the pieces stay in the input's L1 (the capture saw them in DRAM, after the
# model had moved them), so that is what is checked here.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.split`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.split
_NAME = "ttnn.split"

CASES = [
    case(
        "small_4way_32x4096",
        _NAME,
        [t([1, 1, 32, 4096], mem=S.l1()), lit(1024)],
        {"dim": lit(3)},
        [
            out([1, 1, 32, 1024], mem=S.l1()),
            out([1, 1, 32, 1024], mem=S.l1()),
            out([1, 1, 32, 1024], mem=S.l1()),
            out([1, 1, 32, 1024], mem=S.l1()),
        ],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_split_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
