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
Per-op test: ``ttnn.concat`` — every distinct call the model made, as captured.

Captured 7 call(s) to this op; 5 distinct signature(s) covering 7 of them.

Fidelity notes for this op:
  * one argument is a python list of tensors; that repr carries shape/dtype/layout but NOT memory configs, so the list elements are uploaded as DRAM interleaved

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

_OP = ttnn.concat

CASES = [
    {
        "id": "00_32x64x256_bf8_int-dram",
        "op": "ttnn.concat",
        "count": 2,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 32, 64, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 32, 64, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 32, 64, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 32, 64, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 32, 64, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 3},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 1280],
            },
        ],
    },
    {
        "id": "01_64x128x64_bf8_int-dram",
        "op": "ttnn.concat",
        "count": 2,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 64, 128, 64],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 64, 128, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 3},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 320],
            },
        ],
    },
    {
        "id": "02_128x256x32_bf8_int-dram",
        "op": "ttnn.concat",
        "count": 1,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 128, 256, 32],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 128, 256, 128],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 3},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 160],
            },
        ],
    },
    {
        "id": "03_128x256x32_bf8_int-dram",
        "op": "ttnn.concat",
        "count": 1,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 128, 256, 32],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 128, 256, 256],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 3},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 288],
            },
        ],
    },
    {
        "id": "04_128x256x64_bf8_int-dram",
        "op": "ttnn.concat",
        "count": 1,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 128, 256, 64],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 128, 256, 64],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 3},
        },
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
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_concat(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
