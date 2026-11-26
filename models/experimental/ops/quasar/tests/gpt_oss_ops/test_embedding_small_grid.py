# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_embedding.py — do NOT regenerate this file
# from a capture.
#
# The captured tables are the gpt-oss token embedding (201088 x 2880, 1.1 GB) and a
# 131072 x 64 table (16 MiB). Neither fits the 2-node Quasar emulator's DRAM budget, and
# the generated cases are outside the ``-m emulator`` subset for that reason (the suite's
# input-byte cap is 16 MiB).
#
# These variants keep everything about each captured call except the vocabulary size:
# the id shapes, the row-major bfloat16 table, the embedding width (64 and 2880), the
# TILE output layout and the memory configs. The vocabulary is cut to 2048 / 1024 rows
# (4 MiB and 5.6 MiB tables). Token ids are drawn below the table's row count by
# graph_case.INDEX_VALUES, so the gather is checked against the torch golden unchanged.
# The captured output dtype is bfloat8_b, which Quasar has no format for, so the
# variants ask for bfloat16 directly and need no TTNN_GRAPH_OPS_BF8_AS_BF16.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.embedding``: the captured calls on a reduced vocabulary."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.embedding

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _case(case_id, ids_shape, vocab, width, out_shape):
    return {
        "id": case_id,
        "op": "ttnn.embedding",
        "count": 1,
        "args": [
            {"k": "t", "shape": ids_shape, "dtype": "UINT32", "layout": "ROW_MAJOR", "mem": dict(_DRAM)},
            {"k": "t", "shape": [1, 1, vocab, width], "dtype": "BFLOAT16", "layout": "ROW_MAJOR", "mem": dict(_DRAM)},
        ],
        "kwargs": {
            "layout": {"k": "layout", "v": "TILE"},
            "memory_config": {**_DRAM, "k": "mem"},
        },
        "outs": [
            {"dtype": "BFLOAT16", "k": "t", "layout": "TILE", "mem": dict(_DRAM), "shape": out_shape},
        ],
    }


CASES = [
    _case("00_1x32_u32_vocab2048x64_int-dram", [1, 32], 2048, 64, [1, 32, 64]),
    _case("01_1x1_u32_vocab1024x2880_int-dram", [1, 1, 1, 1], 1024, 2880, [1, 1, 2880]),
    _case("02_1x128_u32_vocab1024x2880_int-dram", [1, 1, 1, 128], 1024, 2880, [1, 128, 2880]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_embedding_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
