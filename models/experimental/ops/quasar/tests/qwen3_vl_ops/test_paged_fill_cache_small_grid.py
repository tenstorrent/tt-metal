# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_paged_fill_cache.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call fills a 2784-token prefill (8 KV heads, bfloat8_b) into a 1024-page cache.
# This keeps the page geometry in bfloat16 with 2 KV heads x 64 tokens (two pages, at the
# emulator's 128-row limit) and an 8-page cache; graph_case checks every token landed in the
# page the page table maps.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.experimental.paged_fill_cache`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, lit

_OP = ttnn.experimental.paged_fill_cache
_NAME = "ttnn.experimental.paged_fill_cache"

CASES = [
    case(
        "small_2kv_64tok",
        _NAME,
        [t([8, 2, 32, 128]), t([1, 2, 64, 128]), t([1, 4], dtype="INT32", layout="ROW_MAJOR")],
        {"batch_idx": lit(0)},
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_paged_fill_cache_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
