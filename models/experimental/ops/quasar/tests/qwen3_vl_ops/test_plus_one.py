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
Per-op test: ``ttnn.plus_one`` — every distinct call the model made, as captured.

Captured 800 call(s) to this op; 2 distinct signature(s) covering 800 of them.

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
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

_OP = ttnn.plus_one

CASES = [
    {
        "id": "00_1_i32_int-dram",
        "op": "ttnn.plus_one",
        "count": 400,
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
            "skip_negative_entries": {"k": "lit", "v": True},
        },
        "outs": [
            None,
        ],
    },
    {
        "id": "01_1_u32_int-dram",
        "op": "ttnn.plus_one",
        "count": 400,
        "args": [
            {
                "k": "t",
                "shape": [1],
                "dtype": "UINT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            None,
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_plus_one(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
