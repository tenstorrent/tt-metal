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
Per-op test: ``ttnn.reshard`` — every distinct call the model made, as captured.

Captured 1 call(s) to this op; 1 distinct signature(s) covering 1 of them.

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

_OP = ttnn.reshard

CASES = [
    {
        "id": "00_2048x1024_bf8_bs-l1",
        "op": "ttnn.reshard",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "BLOCK_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 4, 3]], "shape": [512, 224], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 4, 2], [0, 3, 0, 3]], "shape": [128, 1024], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "HEIGHT_SHARDED",
                    "shard": {"grid": [[0, 0, 4, 2], [0, 3, 0, 3]], "orientation": "ROW_MAJOR", "shape": [128, 1024]},
                },
                "shape": [1, 1, 2048, 1024],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reshard(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
