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
Per-op test: ``ttnn.concat`` — every distinct call the model made, as captured.

Captured 21 call(s) to this op; 1 distinct signature(s) covering 21 of them.

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
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.concat

CASES = [
    {
        "id": "00_32x8192_bf8_host",
        "op": "ttnn.concat",
        "count": 21,
        "args": [
            {
                "k": "tlist",
                "tensors": [
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 8192], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                    {"k": "t", "shape": [1, 1, 32, 5376], "dtype": "BFLOAT8_B", "layout": "TILE", "mem": None},
                ],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": -1},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 128256],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_concat(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
