# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_paged_fill_cache.py — do NOT regenerate this
# file from a capture.
#
# The captured call fills a 64-page paged KV cache ([64, 8, 64, 64]) with an
# [1, 8, 128, 64] prefill, through a [1, 2] page table. The tensor being written has
# 1024 rows (suite cap for ``-m emulator``: 128), so the generated case is outside the
# emulator subset.
#
# This variant keeps the call (batch_idx 0, an int32 page table [1, 2], two pages filled
# in sequence, interleaved DRAM, TILE bfloat16) and shrinks the cache to 4 pages of
# [2 heads, 32 slots, 64] with a [1, 2, 64, 64] prefill: 128 rows, and the sequence still
# spans two blocks, so the page-table walk is exercised. graph_case fills the written
# tensor with a sentinel and checks it landed in the pages the (arange) page table maps and
# nowhere else (graph_case.POSTCONDITION).
# ---------------------------------------------------------------------------
"""Small (2-node) variant of ``ttnn.experimental.paged_fill_cache``: the captured call on a reduced cache."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.experimental.paged_fill_cache

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}

CASES = [
    {
        "id": "00_4x2x32x64_bf16_int-dram",
        "op": "ttnn.experimental.paged_fill_cache",
        "count": 1,
        "args": [
            {"k": "t", "shape": [4, 2, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": dict(_DRAM)},
            {"k": "t", "shape": [1, 2, 64, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": dict(_DRAM)},
            {"k": "t", "shape": [1, 2], "dtype": "INT32", "layout": "ROW_MAJOR", "mem": dict(_DRAM)},
        ],
        "kwargs": {"batch_idx": {"k": "lit", "v": 0}},
        "outs": [None],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_paged_fill_cache_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
