# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json --out models/experimental/llama32_1b_quasar/tests/graph_ops
# Source capture: generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.reshape`` — every distinct call the model made, as captured.

Captured 711 call(s) to this op; 5 distinct signature(s) covering 711 of them.

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.reshape

CASES = [
    {
        "id": "00_32x3072_bf16_int-l1",
        "op": "ttnn.reshape",
        "count": 308,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 3072],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 1, 3072)},
            {"k": "lit", "v": (1, 1, 32, 3072)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 3072],
            },
        ],
    },
    {
        "id": "01_32x2048_bf16_ws-l1",
        "op": "ttnn.reshape",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 32], "orientation": "ROW_MAJOR"},
                },
            },
            {"k": "lit", "v": (1, 1, 32, 2048)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 7]], "orientation": "ROW_MAJOR", "shape": [32, 32]},
                },
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "02_1024x2048_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 1, 1024, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1024, 2048],
            },
        ],
    },
    {
        "id": "03_32x1024x64_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 1024, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 32, -1, 64]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 1024, 64],
            },
        ],
    },
    {
        "id": "04_1024x2048_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 1024, 2048)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1024, 2048],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reshape(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
