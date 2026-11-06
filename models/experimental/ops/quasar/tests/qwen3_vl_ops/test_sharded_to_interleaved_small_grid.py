# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_sharded_to_interleaved.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call width-shards [1,1,32,6144] over 39 cores. These width-shard over 1 and
# 2 cores and keep the captured L1-interleaved bfloat16 destination.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.sharded_to_interleaved`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, dtype, mem_arg

_OP = ttnn.sharded_to_interleaved
_NAME = "ttnn.sharded_to_interleaved"

CASES = [
    case(
        "small_ws1_32x128",
        _NAME,
        [t([1, 1, 32, 128], mem=S.sharded("WIDTH_SHARDED", [32, 128])), mem_arg(S.l1()), dtype("BFLOAT16")],
        {},
        [out([1, 1, 32, 128], mem=S.l1())],
    ),
    case(
        "small_ws2_32x256",
        _NAME,
        [t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)), mem_arg(S.l1()), dtype("BFLOAT16")],
        {},
        [out([1, 1, 32, 256], mem=S.l1())],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_sharded_to_interleaved_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
