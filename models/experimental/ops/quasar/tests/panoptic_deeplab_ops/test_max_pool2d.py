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
Per-op test: ``ttnn.max_pool2d`` — every distinct call the model made, as captured.

Captured 2 call(s) to this op; 1 distinct signature(s) covering 2 of them.

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

_OP = ttnn.max_pool2d

CASES = [
    {
        "id": "00_131072x64_bf8_int-dram",
        "op": "ttnn.max_pool2d",
        "count": 2,
        "args": [],
        "kwargs": {
            "input_tensor": {
                "k": "t",
                "shape": [1, 1, 131072, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "channels": {"k": "lit", "v": 64},
            "batch_size": {"k": "lit", "v": 1},
            "input_h": {"k": "lit", "v": 256},
            "input_w": {"k": "lit", "v": 512},
            "kernel_size": {"k": "lit", "v": (3, 3)},
            "stride": {"k": "lit", "v": (2, 2)},
            "padding": {"k": "lit", "v": (1, 1)},
            "dilation": {"k": "lit", "v": (1, 1)},
            "ceil_mode": {"k": "lit", "v": False},
            "deallocate_input": {"k": "lit", "v": False},
            "reallocate_halo_output": {"k": "lit", "v": True},
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "output_layout": {"k": "layout", "v": "TILE"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_max_pool2d(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
