# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_concat.py — do NOT regenerate this file
# from a capture.
#
# The captured case concatenates 15 x [1,1,32,8192] + 1 x [1,1,32,5376] on dim=-1
# into a [1,1,32,128256] (LM-head-width) L1 output = ~8.2 MB, which OOMs the L1 of
# the small-device configs this suite runs on (needs ~4.1 MB/bank across 2 banks vs
# a ~3.8 MB bank). Nothing about the concat path itself needs that width.
#
# This case keeps the same shape of the problem — several equal last-dim tiles plus
# one ragged tail, TILE bf16, dim=-1, L1-interleaved output — but at a tiny width
# (8 x 512 + 256 = 4352 -> 32x4352 output = ~0.27 MB), so it fits any grid while
# still exercising the multi-input last-dim concat.
# ---------------------------------------------------------------------------
"""Small-input companion of ``ttnn.concat`` (multi-input last-dim concat that fits a small grid)."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.concat


def _t(width):
    return {"k": "t", "shape": [1, 1, 32, width], "dtype": "BFLOAT16", "layout": "TILE", "mem": None}


CASES = [
    {
        "id": "small_8x512+256_32_bf16_l1",
        "op": "ttnn.concat",
        "count": 1,
        "args": [
            {
                "k": "tlist",
                "tensors": [_t(512) for _ in range(8)] + [_t(256)],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": -1},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 4352],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_concat_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
