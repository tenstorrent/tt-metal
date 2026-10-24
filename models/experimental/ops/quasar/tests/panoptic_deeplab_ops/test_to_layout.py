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
Per-op test: ``ttnn.to_layout`` — every distinct call the model made, as captured.

Captured 24 call(s) to this op; 14 distinct signature(s) covering 24 of them.

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

_OP = ttnn.to_layout

CASES = [
    {
        "id": "00_1x256_bf16_int-dram",
        "op": "ttnn.to_layout",
        "count": 6,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "TILE"},
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 256],
            },
        ],
    },
    {
        "id": "01_32x64x256_bf16_int-dram",
        "op": "ttnn.to_layout",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "TILE"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 256],
            },
        ],
    },
    {
        "id": "02_64x128x256_bf16_int-dram",
        "op": "ttnn.to_layout",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "TILE"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 256],
            },
        ],
    },
    {
        "id": "03_1x256_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 256],
            },
        ],
    },
    {
        "id": "04_32x64x256_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
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
        "id": "05_64x128x320_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 320],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 320],
            },
        ],
    },
    {
        "id": "06_128x256x128_bf16_int-dram",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 128],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "TILE"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 128],
            },
        ],
    },
    {
        "id": "07_128x256x256_bf16_int-dram",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "TILE"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 256],
            },
        ],
    },
    {
        "id": "08_32x128x256_bf16_int-l1",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 128, 256],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "layout", "v": "TILE"},
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 128, 256],
            },
        ],
    },
    {
        "id": "09_128x256x160_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 160],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 160],
            },
        ],
    },
    {
        "id": "10_128x256x288_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 288],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 288],
            },
        ],
    },
    {
        "id": "11_64x128x128_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 128],
            },
        ],
    },
    {
        "id": "12_64x128x256_bf8_int-dram",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 64, 128, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
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
        "id": "13_128x256x32_bf8_int-l1",
        "op": "ttnn.to_layout",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 256, 32],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "layout", "v": "ROW_MAJOR"},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 32],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_to_layout(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
