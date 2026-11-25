# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_nlp_concat_heads.py — do NOT regenerate this
# file from a capture.
#
# The captured call merges 64 attention heads of a 128-token prefill ([1, 64, 128, 64]
# -> [1, 1, 128, 4096]). Its activation has 8192 rows, so it is outside the ``-m emulator``
# subset (128-row cap).
#
# These variants keep the call (TILE bfloat16, interleaved DRAM, head_dim 64, the same
# memory_config) and cut the head and token counts to 4 heads x 32 tokens and
# 2 heads x 64 tokens (128 rows each). The torch golden (heads concatenated along the
# last dim) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.experimental.nlp_concat_heads``: the captured head merge, reduced."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.experimental.nlp_concat_heads

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}
_HEAD_DIM = 64


def _case(case_id, heads, tokens):
    return {
        "id": case_id,
        "op": "ttnn.experimental.nlp_concat_heads",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, heads, tokens, _HEAD_DIM],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": dict(_DRAM),
            },
        ],
        "kwargs": {"memory_config": {**_DRAM, "k": "mem"}},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": dict(_DRAM),
                "shape": [1, 1, tokens, heads * _HEAD_DIM],
            },
        ],
    }


CASES = [
    _case("00_4x32x64_bf16_int-dram", 4, 32),
    _case("01_2x64x64_bf16_int-dram", 2, 64),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_nlp_concat_heads_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
