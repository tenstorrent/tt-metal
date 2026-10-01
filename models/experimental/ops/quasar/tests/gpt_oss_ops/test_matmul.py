# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/gpt_oss_demo_aug28_1452/graph_capture.python_io.slim.json --out models/experimental/ops/quasar/tests/gpt_oss_ops
# Source capture: generated/ttnn/reports/gpt_oss_demo_aug28_1452/graph_capture.python_io.slim.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.matmul`` — every distinct call the model made, as captured.

Captured 250 call(s) to this op; 5 distinct signature(s) covering 250 of them.

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.matmul

CASES = [
    {
        "id": "00_1x2880_bf8_int-l1",
        "op": "ttnn.matmul",
        "count": 184,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 2880, 5120],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 5120],
            },
        ],
    },
    {
        "id": "01_128x4096_bf8_int-dram",
        "op": "ttnn.matmul",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 4096],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [4096, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 128, 2880],
            },
        ],
    },
    {
        "id": "02_1x2880_bf8_int-dram",
        "op": "ttnn.matmul",
        "count": 8,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 2880, 5120],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 5120],
            },
        ],
    },
    {
        "id": "03_1x2880_bf8_int-l1",
        "op": "ttnn.matmul",
        "count": 8,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "k": "t",
                "shape": [2880, 262144],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            None,
        ],
    },
    {
        "id": "04_32x2880_bf8_int-dram",
        "op": "ttnn.matmul",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [2880, 262144],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 262144],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_matmul(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
