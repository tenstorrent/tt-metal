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
Per-op test: ``ttnn.to_memory_config`` — every distinct call the model made, as captured.

Captured 176230 call(s) to this op; 17 distinct signature(s) covering 176230 of them.

Fidelity notes for this op:
  * an input's logical shape does not fill its shard (e.g. 8 rows in a 32-row shard), so it is built interleaved and relaid out — handing that memory config straight to from_torch would pad the logical shape up to the shard and change what the op computes

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

_OP = ttnn.to_memory_config

CASES = [
    {
        "id": "00_32x2560_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
        "count": 29200,
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
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "01_32x128_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 0, 0]], "orientation": "ROW_MAJOR", "shape": [32, 128]},
                },
                "shape": [1, 1, 32, 128],
            },
        ],
    },
    {
        "id": "02_32x128_bf16_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                },
            },
            {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 128],
            },
        ],
    },
    {
        "id": "03_8x128_bf16_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                },
            },
            {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 8, 128],
            },
        ],
    },
    {
        "id": "04_32x128_bf16_int-l1",
        "op": "ttnn.to_memory_config",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                "k": "mem",
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
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 0, 0]], "orientation": "ROW_MAJOR", "shape": [32, 128]},
                },
                "shape": [1, 1, 32, 128],
            },
        ],
    },
    {
        "id": "05_8x128_bf16_int-l1",
        "op": "ttnn.to_memory_config",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                "k": "mem",
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
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 0, 0]], "orientation": "ROW_MAJOR", "shape": [32, 128]},
                },
                "shape": [1, 1, 8, 128],
            },
        ],
    },
    {
        "id": "06_32x2560_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
                    "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 160], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "07_32x2560_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
                    "shard": {"grid": [[0, 0, 7, 2], [0, 3, 2, 3]], "shape": [32, 96], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "08_32x2560_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 160], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 1]], "orientation": "ROW_MAJOR", "shape": [32, 160]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "09_32x2560_bf16_ws-l1",
        "op": "ttnn.to_memory_config",
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
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
            {"k": "lit", "v": None},
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
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "10_32x9728_bf8_ws-l1",
        "op": "ttnn.to_memory_config",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 9728],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 608], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 608], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 1]], "orientation": "ROW_MAJOR", "shape": [32, 608]},
                },
                "shape": [1, 1, 32, 9728],
            },
        ],
    },
    {
        "id": "11_32x26720_bf8_ws-l1",
        "op": "ttnn.to_memory_config",
        "count": 2010,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 26720],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 672], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            None,
        ],
    },
    {
        "id": "12_32x18336_bf8_ws-l1",
        "op": "ttnn.to_memory_config",
        "count": 402,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 18336],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 3], [0, 4, 6, 4]], "shape": [32, 480], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            None,
        ],
    },
    {
        "id": "13_32x2560_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "14_4096x2560_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 144,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
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
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "15_4096x2560_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
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
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "16_32x2560_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
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
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_to_memory_config(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
