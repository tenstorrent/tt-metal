# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_nlp_concat_heads_decode.py — do NOT
# regenerate this file from a capture.
#
# The captured case (num_heads=32) width-shards its output one head per core, so
# it needs 32 compute cores (nlp_concat_heads_decode_device_operation.cpp: the
# output grid is num_cores_to_corerangeset(num_heads, grid)). On a 2-compute-core
# device it FATALs in work_split.cpp:98 (target_num_cores 32 > 2 available).
#
# This case sets num_heads=2, so the output width-shards onto exactly 2 cores and
# fits a 2-compute-core grid, while still exercising the real multi-core width-
# sharded path. The op only requires input_shape[2] >= num_heads, so the same
# tile-aligned [1,1,32,64] input is reused. The derived output grid is cores
# (0,0),(1,0) => [[0,0,1,0]] on any grid >= 2 cores, so this also passes on a full
# grid unchanged.
# ---------------------------------------------------------------------------
"""Small-grid smoke test for ``ttnn.experimental.nlp_concat_heads_decode`` (num_heads=2)."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.experimental.nlp_concat_heads_decode

CASES = [
    {
        "id": "small_2heads_2x64_bf16_hs-l1",
        "op": "ttnn.experimental.nlp_concat_heads_decode",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "num_heads": {"k": "lit", "v": 2},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 1, 0]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 128],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_nlp_concat_heads_decode_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
