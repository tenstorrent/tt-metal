# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_paged_update_cache.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call writes one decode token's 8 KV heads ([1,1,8,128], height-sharded on one
# core) into a 1024-page bfloat8_b cache (34 MB). This keeps the update and page geometry
# with an 8-page bfloat16 cache; graph_case checks the write landed where the page table says.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.experimental.paged_update_cache`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t

_OP = ttnn.experimental.paged_update_cache
_NAME = "ttnn.experimental.paged_update_cache"

CASES = [
    case(
        "small_8pages_8x32x128",
        _NAME,
        [t([8, 8, 32, 128]), t([1, 1, 8, 128], mem=S.sharded("HEIGHT_SHARDED", [32, 128]))],
        {
            "update_idxs_tensor": t([1], dtype="INT32", layout="ROW_MAJOR"),
            "page_table": t([1, 8], dtype="INT32", layout="ROW_MAJOR"),
        },
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_paged_update_cache_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
