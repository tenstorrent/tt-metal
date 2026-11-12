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
Per-op test: ``ttnn.add`` — every distinct call the model made, as captured.

Captured 36 call(s) to this op; 8 distinct signature(s) covering 36 of them.

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

_OP = ttnn.add

CASES = [
    {
        "id": "00_2048x256_bf8_bs-l1",
        "op": "ttnn.add",
        "count": 14,
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
            {
                "k": "t",
                "shape": [1, 1, 2048, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "output_tensor": {
                "k": "t",
                "shape": [1, 1, 2048, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        },
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
        "id": "01_2048x256_bf8_int-dram",
        "op": "ttnn.add",
        "count": 6,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 1, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "output_tensor": {
                "k": "t",
                "shape": [1, 1, 2048, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        },
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
        "id": "02_2048x1024_bf8_hs-l1",
        "op": "ttnn.add",
        "count": 6,
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
        ],
        "kwargs": {
            "activations": {"k": "uwp", "v": [{"op": "RELU", "params": []}], "list": True},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 4, 2], [0, 3, 0, 3]], "orientation": "ROW_MAJOR", "shape": [128, 1024]},
                },
                "shape": [1, 1, 2048, 1024],
            },
        ],
    },
    {
        "id": "03_2048x2048_bf8_int-dram",
        "op": "ttnn.add",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "activations": {"k": "uwp", "v": [{"op": "RELU", "params": []}], "list": True},
        },
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
        "id": "04_32768x256_bf8_int-dram",
        "op": "ttnn.add",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 32768, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "activations": {"k": "uwp", "v": [{"op": "RELU", "params": []}], "list": True},
        },
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
        "id": "05_8192x512_bf8_hs-l1",
        "op": "ttnn.add",
        "count": 2,
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
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "activations": {"k": "uwp", "v": [{"op": "RELU", "params": []}], "list": True},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 4, 3]], "orientation": "ROW_MAJOR", "shape": [416, 512]},
                },
                "shape": [1, 1, 8192, 512],
            },
        ],
    },
    {
        "id": "06_8192x512_bf8_int-dram",
        "op": "ttnn.add",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "activations": {"k": "uwp", "v": [{"op": "RELU", "params": []}], "list": True},
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
        "id": "07_8192x512_bf8_hs-l1",
        "op": "ttnn.add",
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
            "activations": {"k": "uwp", "v": [{"op": "RELU", "params": []}], "list": True},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 4, 3]], "orientation": "ROW_MAJOR", "shape": [416, 512]},
                },
                "shape": [1, 1, 8192, 512],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_add(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
