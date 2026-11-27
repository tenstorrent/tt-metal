# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_fast_reduce_nc.py — do NOT regenerate this file
# from a capture.
#
# The captured call is the gpt-oss expert combine (reduce_experts): a sum over the
# expert dim of [1, 32 experts, 128 tokens, 2880], dims=[1]. That input is 22 MiB as
# bfloat16 (the suite runs the captured bfloat8_b as bfloat16) with a 4096-row activation, so
# it is outside the ``-m emulator`` subset (128-row cap).
#
# These variants keep the call (dims=[1], TILE bfloat16, interleaved DRAM, the model's
# 2880 width) and cut the expert and token counts: 4 experts x 32 tokens (128 rows).
# The torch golden (sum over dims, kept) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.experimental.fast_reduce_nc``: the captured expert-dim sum, reduced."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.experimental.fast_reduce_nc

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _case(case_id, experts, tokens, width):
    return {
        "id": case_id,
        "op": "ttnn.experimental.fast_reduce_nc",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, experts, tokens, width],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": dict(_DRAM),
            },
        ],
        "kwargs": {"dims": {"k": "lit", "v": [1]}},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": dict(_DRAM),
                "shape": [1, 1, tokens, width],
            },
        ],
    }


CASES = [
    _case("00_4x32x2880_bf16_int-dram", 4, 32, 2880),
    _case("01_2x64x2880_bf16_int-dram", 2, 64, 2880),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fast_reduce_nc_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
