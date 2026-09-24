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
Per-op test: ``ttnn.Tensor.__getitem__`` — every distinct call the model made, as captured.

Captured 108 call(s) to this op; 3 distinct signature(s) covering 108 of them.

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

_OP = ttnn.Tensor.__getitem__

CASES = [
    {
        "id": "00_8x1024x64_bf8_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 64,
        "args": [
            {
                "k": "t",
                "shape": [1, 8, 1024, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, None, None], [None, 512, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 8, 512, 64],
            },
        ],
    },
    {
        "id": "01_32x1x64_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 40,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 1, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, 1, None], [None, None, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 64],
            },
        ],
    },
    {
        "id": "02_8192x64_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 4,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, None, None], [0, 1024, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1024, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_tensor_getitem(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
