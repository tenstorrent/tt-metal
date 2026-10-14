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
Per-op test: ``ttnn.upsample`` — every distinct call the model made, as captured.

Captured 6 call(s) to this op; 4 distinct signature(s) covering 6 of them.

Fidelity notes for this op:
  * the output has more elements than all inputs combined (a batch-padded decode tensor), so the op cannot write all of it; finiteness is asserted over the portion the inputs can account for

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

_OP = ttnn.upsample

CASES = [
    {
        "id": "00_32x64x256_bf16_int-dram",
        "op": "ttnn.upsample",
        "count": 2,
        "args": [],
        "kwargs": {
            "input_tensor": {
                "k": "t",
                "shape": [1, 32, 64, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "scale_factor": {"k": "lit", "v": [2, 2]},
            "mode": {"k": "lit", "v": "nearest"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 256],
            },
        ],
    },
    {
        "id": "01_1x256_bf16_int-dram",
        "op": "ttnn.upsample",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "scale_factor": {"k": "lit", "v": (32, 64)},
            "mode": {"k": "lit", "v": "nearest"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 256],
            },
        ],
    },
    {
        "id": "02_64x128x256_bf16_int-dram",
        "op": "ttnn.upsample",
        "count": 1,
        "args": [],
        "kwargs": {
            "input_tensor": {
                "k": "t",
                "shape": [1, 64, 128, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "scale_factor": {"k": "lit", "v": [2, 2]},
            "mode": {"k": "lit", "v": "nearest"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 256],
            },
        ],
    },
    {
        "id": "03_64x128x128_bf16_int-dram",
        "op": "ttnn.upsample",
        "count": 1,
        "args": [],
        "kwargs": {
            "input_tensor": {
                "k": "t",
                "shape": [1, 64, 128, 128],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "scale_factor": {"k": "lit", "v": [2, 2]},
            "mode": {"k": "lit", "v": "nearest"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 128],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_upsample(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
