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
Per-op test: ``ttnn.to_memory_config`` — every distinct call the model made, as captured.

Captured 576 call(s) to this op; 2 distinct signature(s) covering 576 of them.

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
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.to_memory_config

CASES = [
    {
        "id": "00_8x64_bf16_hs-l1",
        "op": "ttnn.to_memory_config",
        "count": 384,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 0, 0]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 8, 64],
            },
        ],
    },
    {
        "id": "01_64x64_bf16_int-dram",
        "op": "ttnn.to_memory_config",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 64, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [64, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 0, 0]], "orientation": "ROW_MAJOR", "shape": [64, 64]},
                },
                "shape": [1, 1, 64, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_to_memory_config(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
