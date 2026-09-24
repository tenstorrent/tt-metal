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
Per-op test: ``ttnn.to_memory_config`` — every distinct call the model made, as captured.

Captured 2531 call(s) to this op; 11 distinct signature(s) covering 2531 of them.

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

_OP = ttnn.to_memory_config

CASES = [
    {
        "id": "00_32x2048_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 327,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 3]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "01_32x2048_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 308,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            {"k": "lit", "v": None},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "02_32x2048_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 32], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        },
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
        "id": "03_32x64_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 0, 0]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 64],
            },
        ],
    },
    {
        "id": "04_32x2048_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
                    "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "05_32x2048_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
                    "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
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
                    "shard": {"grid": [[0, 0, 7, 3]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "06_32x2048_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
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
                    "shard": {"grid": [[0, 0, 7, 3]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "07_32x8192_bf8_ws-l1",
        "op": "ttnn.to_memory_config",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 8192],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        ],
        "kwargs": {},
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
        "id": "08_1024x2048_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
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
        "id": "09_32x2048_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 20,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "10_32x128256_bf8_int-l1",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 128256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 128256],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_to_memory_config(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
