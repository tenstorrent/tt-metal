# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_nlp_concat_heads_decode.py — do NOT
# regenerate this file from a capture.
#
# The captured case (num_heads=64) width-shards its output one head per core, so
# it needs 64 compute cores (nlp_concat_heads_decode_device_operation.cpp: the
# output grid is num_cores_to_corerangeset(num_heads, grid)). It FATALs in
# work_split.cpp ("Target number of cores 64 is greater than total number of
# available cores") on a 2-core grid, and also on the full 8x4 Quasar grid (32
# cores).
#
# Two kinds of test cover the small grid:
#
#   * test_nlp_concat_heads_decode_small_grid runs the op itself with num_heads=2,
#     so the output width-shards onto exactly 2 cores and the op's own multi-core
#     width-sharded path runs on a 2-core grid. The input keeps the captured dtype,
#     layout, height-sharded SRAM placement, batch 1 and head_dim 64; only the head
#     count changes, to [1,1,2,64] in one 32x64 shard. Because the input has
#     exactly num_heads heads, the golden reference in graph_case applies and the
#     values are compared with torch. The derived output grid is cores
#     (0,0),(1,0) => [[0,0,1,0]] on any grid with at least 2 columns.
#
#   * test_merge_heads_on_device runs the captured 64-head case unchanged through
#     merge_heads_on_device from models/experimental/ops/quasar/gpt_oss/conftest.py.
#     That is what the gpt-oss model runs in place of nlp_concat_heads_decode on a
#     device with fewer cores than heads (the quasar_ops fixture in that conftest).
#     merge_heads_on_device returns its output interleaved in DRAM, not width-
#     sharded in SRAM as the op does, and graph_case checks the output placement
#     before the values, so the expected output placement of this case is
#     interleaved DRAM. Shape, dtype, layout and values are checked as captured.
# ---------------------------------------------------------------------------
"""Small-grid companion of ``ttnn.experimental.nlp_concat_heads_decode`` (num_heads=2, and the gpt-oss head merge)."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.gpt_oss.conftest import merge_heads_on_device
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops.test_nlp_concat_heads_decode import CASES as CAPTURED_CASES

_OP = ttnn.experimental.nlp_concat_heads_decode

CASES = [
    {
        "id": "small_2heads_2x64_bf16_hs-l1",
        "op": "ttnn.experimental.nlp_concat_heads_decode",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2, 64],
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


def _merge_case(captured):
    """The captured case, with the output placement that merge_heads_on_device produces."""
    case = copy.deepcopy(captured)
    case["id"] = f"{captured['id']}_merge-dram"
    for out in case["outs"]:
        out["mem"] = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}
    return case


MERGE_CASES = [_merge_case(c) for c in CAPTURED_CASES]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_nlp_concat_heads_decode_small_grid(ttnn_mesh_device, reset_seeds, case):
    num_heads = case["kwargs"]["num_heads"]["v"]
    dev = ttnn_mesh_device.compute_with_storage_grid_size()
    if dev.x * dev.y < num_heads:
        pytest.skip(f"case needs {num_heads} compute cores (one per head); device compute grid is {dev.x}x{dev.y}")
    G.run_case(_OP, case, ttnn_mesh_device)


@G.with_default_mesh()
@pytest.mark.parametrize("case", MERGE_CASES, ids=[c["id"] for c in MERGE_CASES])
def test_merge_heads_on_device(ttnn_mesh_device, reset_seeds, case):
    G.run_case(merge_heads_on_device, case, ttnn_mesh_device)
