# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/qwen3_vl_demo_aug27_1509/graph_capture.python_io.slim.json --out models/experimental/ops/quasar/tests/qwen3_vl_ops
# Source capture: generated/ttnn/reports/qwen3_vl_demo_aug27_1509/graph_capture.python_io.slim.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.add`` — every distinct call the model made, as captured.

Captured 29238 call(s) to this op; 9 distinct signature(s) covering 29238 of them.

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

_OP = ttnn.add

CASES = [
    {
        "id": "00_32x2560_bf16_ws-l1",
        "op": "ttnn.add",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
            "dtype": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "01_32x2560_bf16_ws-l1",
        "op": "ttnn.add",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "02_12288x1024_bf16_int-dram",
        "op": "ttnn.add",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 12288, 1024],
            },
        ],
    },
    {
        "id": "03_12288x1024_bf16_int-dram",
        "op": "ttnn.add",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "dtype": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 12288, 1024],
            },
        ],
    },
    {
        "id": "04_4096x2560_bf16_int-dram",
        "op": "ttnn.add",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "dtype": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "05_4096x2560_bf16_int-dram",
        "op": "ttnn.add",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "06_6x2048x3072_bf16_int-dram",
        "op": "ttnn.add",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 6, 2048, 3072],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [3072],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 6, 2048, 3072],
            },
        ],
    },
    {
        "id": "07_12x1024x1024_bf8_int-dram",
        "op": "ttnn.add",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 12, 1024, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 12, 1024, 1024],
            },
        ],
    },
    {
        "id": "08_4096x2560_bf16_int-dram",
        "op": "ttnn.add",
        "count": 6,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_add(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
