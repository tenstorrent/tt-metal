# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_add.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured decode adds width-shard [1,1,32,2560] over a 40-core grid; the prefill adds
# are 4096-12288 rows. These keep both forms: width-sharded over 2 cores (with and without
# an explicit dtype, as captured), an interleaved DRAM add, and the two broadcast shapes
# the vision tower uses (a [N] bias row and a 2D [rows, N] operand).
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.add`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit, dtype, mem_arg

# The base binary_ng op still builds Gen1 DataMovementKernels, which Quasar rejects; its
# ttnn.experimental.quasar port takes the same arguments and runs on every arch. _NAME keeps
# the captured op name, so the golden lookup is unchanged.
_OP = ttnn.experimental.quasar.add
_NAME = "ttnn.add"

CASES = [
    case(
        "small_ws2_32x256",
        _NAME,
        [
            t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
            t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
        ],
        {"memory_config": mem_arg(S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)), "dtype": lit(None)},
        [out([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2))],
    ),
    case(
        "small_ws2_32x256_dtype",
        _NAME,
        [
            t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
            t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
        ],
        {"memory_config": mem_arg(S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)), "dtype": dtype("BFLOAT16")},
        [out([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2))],
    ),
    case(
        "small_int_128x512",
        _NAME,
        [t([1, 1, 128, 512]), t([1, 1, 128, 512])],
        {"memory_config": mem_arg(S.dram()), "dtype": dtype("BFLOAT16")},
        [out([1, 1, 128, 512])],
    ),
    case(
        "small_bcast_row_128x512",
        _NAME,
        [t([1, 1, 128, 512]), t([512])],
        {"memory_config": mem_arg(S.dram()), "dtype": dtype("BFLOAT16")},
        [out([1, 1, 128, 512])],
    ),
    case(
        "small_bcast_2d_128x512",
        _NAME,
        [t([1, 1, 128, 512]), t([128, 512])],
        {"memory_config": mem_arg(S.dram()), "dtype": dtype("BFLOAT16")},
        [out([1, 1, 128, 512])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_add_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
