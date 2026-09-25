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
Per-op test: ``ttnn.linear`` — every distinct call the model made, as captured.

Captured 480 call(s) to this op; 4 distinct signature(s) covering 480 of them.

Fidelity notes for this op:
  * compute_kernel_config is recorded only as an object address in the capture, so it is dropped and the op's default is used (can shift PCC, not shapes)

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

_OP = ttnn.linear

CASES = [
    {
        "id": "00_32x4096_bf16_ws-l1",
        "op": "ttnn.linear",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 4096],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 10, 4], [0, 5, 8, 5]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
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
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 10, 3], [0, 4, 0, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2880],
            },
        ],
    },
    {
        "id": "01_1x2880_bf8_int-l1",
        "op": "ttnn.linear",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "k": "t",
                "shape": [2880, 32],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "bias": {
                "k": "t",
                "shape": [1, 32],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32],
            },
        ],
    },
    {
        "id": "02_128x2880_bf16_int-dram",
        "op": "ttnn.linear",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [128, 2880],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [2880, 32],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "bias": {
                "k": "t",
                "shape": [1, 32],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [128, 32],
            },
        ],
    },
    {
        "id": "03_128x2880_bf8_int-dram",
        "op": "ttnn.linear",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 2880],
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
            "bias": {
                "k": "t",
                "shape": [5120],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 128, 5120],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_linear(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
