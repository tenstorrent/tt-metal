# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_nlp_concat_heads_decode.py — do NOT
# regenerate this file from a capture.
#
# The captured case (num_heads=32) width-shards its output one head per core, so
# it needs 32 compute cores (nlp_concat_heads_decode_device_operation.cpp: the
# output grid is num_cores_to_corerangeset(num_heads, grid)); graph_case skips it
# on a smaller device. These cases keep Qwen3-VL's head_dim=128 and the captured
# [1,1,32,128] HEIGHT_SHARDED input layout but set num_heads to 1 and 2, so the
# output lands on 1 and 2 cores: they run on a 1- or 2-compute-node device and,
# with 2 heads, still exercise the multi-core width-sharded path. The op only
# requires input_shape[2] >= num_heads, and the derived output grid starts at
# (0,0) on any grid, so these also pass unchanged on a full grid.
# ---------------------------------------------------------------------------
"""Small-grid tests for ``ttnn.experimental.nlp_concat_heads_decode`` (num_heads=1 and 2)."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

_OP = ttnn.experimental.nlp_concat_heads_decode

_HEAD_DIM = 128  # Qwen3-VL head_dim, as in the captured case


def _case(num_heads):
    return {
        "id": f"small_{num_heads}heads_32x{_HEAD_DIM}_bf16_hs-l1",
        "op": "ttnn.experimental.nlp_concat_heads_decode",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, _HEAD_DIM],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, _HEAD_DIM], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "num_heads": {"k": "lit", "v": num_heads},
            "sub_core_grids": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {
                        "grid": [[0, 0, num_heads - 1, 0]],
                        "orientation": "ROW_MAJOR",
                        "shape": [32, _HEAD_DIM],
                    },
                },
                "shape": [1, 1, 32, num_heads * _HEAD_DIM],
            },
        ],
    }


CASES = [_case(1), _case(2)]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_nlp_concat_heads_decode_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
