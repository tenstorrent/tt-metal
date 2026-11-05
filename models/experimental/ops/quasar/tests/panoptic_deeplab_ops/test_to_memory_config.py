# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/panoptic_deeplab_20_cores_oct14_1821/graph_capture.python_io.json --out models/experimental/ops/quasar/tests/panoptic_deeplab_ops --max-cases 64
# Source capture: generated/ttnn/reports/panoptic_deeplab_20_cores_oct14_1821/graph_capture.python_io.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.to_memory_config`` — every distinct call the model made, as captured.

Captured 61 call(s) to this op; 27 distinct signature(s) covering 61 of them.

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
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.to_memory_config

CASES = [
    {
        "id": "00_32x64x256_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 12,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 256],
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
                "shape": [1, 32, 64, 256],
            },
        ],
    },
    {
        "id": "01_2048x256_bf8_bs-l1",
        "op": "ttnn.to_memory_config",
        "count": 6,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "BLOCK_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 3, 3]], "shape": [512, 64], "orientation": "ROW_MAJOR"},
                },
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
                "shape": [1, 1, 2048, 256],
            },
        ],
    },
    {
        "id": "02_2048x512_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 512],
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
                "shape": [1, 1, 2048, 512],
            },
        ],
    },
    {
        "id": "03_128x256x32_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 32],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 32],
            },
        ],
    },
    {
        "id": "04_32768x64_bf8_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 3]], "shape": [1664, 64], "orientation": "ROW_MAJOR"},
                },
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
                "shape": [1, 1, 32768, 64],
            },
        ],
    },
    {
        "id": "05_1x2048_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2048],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 2048],
            },
        ],
    },
    {
        "id": "06_32x64x256_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 256],
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
                "shape": [1, 32, 64, 256],
            },
        ],
    },
    {
        "id": "07_64x128x256_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 256],
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
                "shape": [1, 64, 128, 256],
            },
        ],
    },
    {
        "id": "08_1x256_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 256],
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
                "shape": [1, 1, 1, 256],
            },
        ],
    },
    {
        "id": "09_32768x32_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 32],
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
                "shape": [1, 1, 32768, 32],
            },
        ],
    },
    {
        "id": "10_128x256x128_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 128],
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
                "shape": [1, 128, 256, 128],
            },
        ],
    },
    {
        "id": "11_128x256x256_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 256],
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
                "shape": [1, 128, 256, 256],
            },
        ],
    },
    {
        "id": "12_128x256x32_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 32],
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
                "shape": [1, 128, 256, 32],
            },
        ],
    },
    {
        "id": "13_128x256x64_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            None,
        ],
    },
    {
        "id": "14_64x128x128_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 128],
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
                "shape": [1, 64, 128, 128],
            },
        ],
    },
    {
        "id": "15_64x128x256_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 256],
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
                "shape": [1, 64, 128, 256],
            },
        ],
    },
    {
        "id": "16_64x128x64_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 64],
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
                "shape": [1, 64, 128, 64],
            },
        ],
    },
    {
        "id": "17_128x256x128_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 128],
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
                "shape": [1, 128, 256, 128],
            },
        ],
    },
    {
        "id": "18_128x256x256_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 256],
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
                "shape": [1, 128, 256, 256],
            },
        ],
    },
    {
        "id": "19_2048x2048_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
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
                "shape": [1, 1, 2048, 2048],
            },
        ],
    },
    {
        "id": "20_32768x256_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 256],
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
                "shape": [1, 1, 32768, 256],
            },
        ],
    },
    {
        "id": "21_8192x512_bf8_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
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
                "shape": [1, 1, 8192, 512],
            },
        ],
    },
    {
        "id": "22_2048x1024_bf8_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 2], [0, 3, 0, 3]], "shape": [128, 1024], "orientation": "ROW_MAJOR"},
                },
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
                "shape": [1, 1, 2048, 1024],
            },
        ],
    },
    {
        "id": "23_8192x128_bf8_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 2], [0, 3, 0, 3]], "shape": [512, 128], "orientation": "ROW_MAJOR"},
                },
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
                "shape": [1, 1, 8192, 128],
            },
        ],
    },
    {
        "id": "24_8192x512_bf8_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 3]], "shape": [416, 512], "orientation": "ROW_MAJOR"},
                },
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
                "shape": [1, 1, 8192, 512],
            },
        ],
    },
    {
        "id": "25_8192x512_bf8_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 3]], "shape": [416, 512], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 8192, 512],
            },
        ],
    },
    {
        "id": "26_131072x64_bf8_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 131072, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 3]], "shape": [6560, 64], "orientation": "ROW_MAJOR"},
                },
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
                "shape": [1, 1, 131072, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_to_memory_config(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
