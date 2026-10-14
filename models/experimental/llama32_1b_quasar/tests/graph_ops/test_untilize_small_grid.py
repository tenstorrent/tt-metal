# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_untilize.py — do NOT regenerate this
# file from a capture.
#
# Both captured cases untilize a [1,1,32,128256] (LM-head-width) tensor. The
# row-major result is ~8.2 MB, which OOMs L1 on the small-device configs this suite
# runs on, so graph_ops/conftest.py skips anything with "32x128256" in its id.
# Nothing about the untilize path needs that width.
#
# These variants keep the same signatures (TILE bf16 input, use_multicore, one with
# an L1 input + DRAM output and one with a DRAM input + default output) but at a tiny
# width (2048 = 64 tiles -> 32x2048 output = ~0.13 MB), so untilize actually runs on
# a small device.
# ---------------------------------------------------------------------------
"""Small-input companion of ``ttnn.untilize`` (fits a small-device L1)."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.untilize

_W = 2048  # 64 tiles wide

CASES = [
    {
        # mirrors 00: L1 interleaved input -> DRAM interleaved output
        "id": "small_32x2048_bf16_int-l1",
        "op": "ttnn.untilize",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, _W],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {
            "use_multicore": {"k": "lit", "v": True},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [None],
    },
    {
        # mirrors 01: DRAM interleaved input -> default output
        "id": "small_32x2048_bf16_int-dram",
        "op": "ttnn.untilize",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, _W],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "use_multicore": {"k": "lit", "v": True},
        },
        "outs": [None],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_untilize_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
