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
Per-op test: ``ttnn.clamp`` — every distinct call the model made, as captured.

Captured 480 call(s) to this op; 4 distinct signature(s) covering 480 of them.

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

_OP = ttnn.clamp

CASES = [
    {
        "id": "00_32x2880_bf8_int-l1",
        "op": "ttnn.clamp",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {
            "min": {"k": "lit", "v": None},
            "max": {"k": "lit", "v": 7.0},
            "output_tensor": {
                "k": "t",
                "shape": [1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 2880],
            },
        ],
    },
    {
        "id": "01_32x2880_bf8_int-l1",
        "op": "ttnn.clamp",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {
            "min": {"k": "lit", "v": -7.0},
            "max": {"k": "lit", "v": 7.0},
            "output_tensor": {
                "k": "t",
                "shape": [1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 2880],
            },
        ],
    },
    {
        "id": "02_32x128x2880_bf8_int-dram",
        "op": "ttnn.clamp",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "min": {"k": "lit", "v": None},
            "max": {"k": "lit", "v": 7.0},
            "output_tensor": {
                "k": "t",
                "shape": [1, 32, 128, 2880],
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
                "shape": [1, 32, 128, 2880],
            },
        ],
    },
    {
        "id": "03_32x128x2880_bf8_int-dram",
        "op": "ttnn.clamp",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "min": {"k": "lit", "v": -7.0},
            "max": {"k": "lit", "v": 7.0},
            "output_tensor": {
                "k": "t",
                "shape": [1, 32, 128, 2880],
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
                "shape": [1, 32, 128, 2880],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_clamp(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
