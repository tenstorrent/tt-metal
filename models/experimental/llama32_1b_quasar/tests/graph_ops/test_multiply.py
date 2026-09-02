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
Per-op test: ``ttnn.multiply`` — every distinct call the model made, as captured.

Captured 339 call(s) to this op; 2 distinct signature(s) covering 339 of them.

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

_OP = ttnn.multiply

CASES = [
    {
        "id": "00_32x8192_bf16_ws-l1",
        "op": "ttnn.multiply",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 8192],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 32, 8192],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "input_tensor_a_activations": {"k": "acts", "v": ["SILU"]},
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 7]], "orientation": "ROW_MAJOR", "shape": [32, 128]},
                },
                "shape": [1, 1, 32, 8192],
            },
        ],
    },
    {
        "id": "01_1024x8192_bf16_int-dram",
        "op": "ttnn.multiply",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 8192],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 1024, 8192],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "input_tensor_a_activations": {"k": "acts", "v": ["SILU"]},
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1024, 8192],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_multiply(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
