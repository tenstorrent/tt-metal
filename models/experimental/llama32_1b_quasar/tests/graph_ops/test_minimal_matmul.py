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
Per-op test: ``ttnn.experimental.minimal_matmul`` — every distinct call the model made, as captured.

Captured 64 call(s) to this op; 2 distinct signature(s) covering 64 of them.

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
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.experimental.minimal_matmul

CASES = [
    {
        "id": "00_1024x2048_bf16_int-dram",
        "op": "ttnn.experimental.minimal_matmul",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 2048, 3072],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [2048, 256], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "config": {
                "kind": "MinimalMatmulConfig",
                "fields": {
                    "M_block_size": 8,
                    "K_block_size": 8,
                    "N_block_size": 8,
                    "subblock_h": 1,
                    "subblock_w": 1,
                    "compute_with_storage_grid_size": [8, 8],
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1024, 3072],
            },
        ],
    },
    {
        "id": "01_1024x8192_bf8_int-dram",
        "op": "ttnn.experimental.minimal_matmul",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 8192],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [8192, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [8192, 192], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "config": {
                "kind": "MinimalMatmulConfig",
                "fields": {
                    "M_block_size": 8,
                    "K_block_size": 8,
                    "N_block_size": 8,
                    "subblock_h": 1,
                    "subblock_w": 1,
                    "compute_with_storage_grid_size": [8, 8],
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1024, 2048],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_minimal_matmul(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
