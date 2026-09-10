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
Per-op test: ``ttnn.to_device`` — every distinct call the model made, as captured.

Captured 80 call(s) to this op; 4 distinct signature(s) covering 80 of them.

Fidelity notes for this op:
  * an integer index tensor is involved; the capture holds no values, so it is filled by graph_case.INDEX_VALUES (page ids, positions, token ids) instead of random data

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

_OP = ttnn.to_device

CASES = [
    {
        "id": "00_1x128_i32_int-dram",
        "op": "ttnn.to_device",
        "count": 20,
        "args": [
            {
                "k": "t",
                "shape": [1, 128],
                "dtype": "INT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "device": {"k": "device"},
        },
        "outs": [
            {
                "dtype": "INT32",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128],
            },
        ],
    },
    {
        "id": "01_1_i32_int-dram",
        "op": "ttnn.to_device",
        "count": 20,
        "args": [
            {
                "k": "t",
                "shape": [1],
                "dtype": "INT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "device": {"k": "device"},
        },
        "outs": [
            {
                "dtype": "INT32",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1],
            },
        ],
    },
    {
        "id": "02_1x32_u32_int-dram",
        "op": "ttnn.to_device",
        "count": 20,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 32],
                "dtype": "UINT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "device": {"k": "device"},
        },
        "outs": [
            {
                "dtype": "UINT32",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 32],
            },
        ],
    },
    {
        "id": "03_1x32_u32_int-dram",
        "op": "ttnn.to_device",
        "count": 20,
        "args": [
            {
                "k": "t",
                "shape": [1, 32],
                "dtype": "UINT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "device": {"k": "device"},
        },
        "outs": [
            {
                "dtype": "UINT32",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_to_device(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
