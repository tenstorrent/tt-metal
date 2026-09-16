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
Per-op test: ``ttnn.concat`` — every distinct call the model made, as captured.

Captured 804 call(s) to this op; 3 distinct signature(s) covering 804 of them.

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
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

_OP = ttnn.concat

CASES = [
    {
        "id": "00_32x26720_bf8_int-l1",
        "op": "ttnn.concat",
        "count": 402,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 26720],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 26720],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 26720],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 26720],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 26720],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 18336],
                        "dtype": "BFLOAT8_B",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": -1},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
            "sub_core_grids": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 151936],
            },
        ],
    },
    {
        "id": "01_32x37984_bf16_int-dram",
        "op": "ttnn.concat",
        "count": 400,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 37984],
                        "dtype": "BFLOAT16",
                        "layout": "ROW_MAJOR",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 37984],
                        "dtype": "BFLOAT16",
                        "layout": "ROW_MAJOR",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 37984],
                        "dtype": "BFLOAT16",
                        "layout": "ROW_MAJOR",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1, 1, 32, 37984],
                        "dtype": "BFLOAT16",
                        "layout": "ROW_MAJOR",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 3},
            "sub_core_grids": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 151936],
            },
        ],
    },
    {
        "id": "02_2766x2560_bf16_int-dram",
        "op": "ttnn.concat",
        "count": 2,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {
                        "k": "t",
                        "shape": [2766, 2560],
                        "dtype": "BFLOAT16",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                    {
                        "k": "t",
                        "shape": [1330, 2560],
                        "dtype": "BFLOAT16",
                        "layout": "TILE",
                        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
                    },
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": 0},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [4096, 2560],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_concat(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
