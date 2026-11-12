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
Per-op test: ``ttnn.slice`` — every distinct call the model made, as captured.

Captured 22 call(s) to this op; 12 distinct signature(s) covering 22 of them.

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

_OP = ttnn.slice

CASES = [
    {
        "id": "00_2048x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 0]},
            {"k": "lit", "v": [1, 1, 2048, 512]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2048, 512],
            },
        ],
    },
    {
        "id": "01_2048x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 1024]},
            {"k": "lit", "v": [1, 1, 2048, 1536]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2048, 512],
            },
        ],
    },
    {
        "id": "02_2048x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 1536]},
            {"k": "lit", "v": [1, 1, 2048, 2048]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2048, 512],
            },
        ],
    },
    {
        "id": "03_2048x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 512]},
            {"k": "lit", "v": [1, 1, 2048, 1024]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2048, 512],
            },
        ],
    },
    {
        "id": "04_32x64x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 0]},
            {"k": "lit", "v": [1, 32, 64, 1024]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 1024],
            },
        ],
    },
    {
        "id": "05_32x64x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 0]},
            {"k": "lit", "v": [1, 32, 64, 512]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 512],
            },
        ],
    },
    {
        "id": "06_32x64x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 1024]},
            {"k": "lit", "v": [1, 32, 64, 1536]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 512],
            },
        ],
    },
    {
        "id": "07_32x64x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 1024]},
            {"k": "lit", "v": [1, 32, 64, 2048]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 1024],
            },
        ],
    },
    {
        "id": "08_32x64x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 1536]},
            {"k": "lit", "v": [1, 32, 64, 2048]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 512],
            },
        ],
    },
    {
        "id": "09_32x64x2048_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 512]},
            {"k": "lit", "v": [1, 32, 64, 1024]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 512],
            },
        ],
    },
    {
        "id": "10_131072x128_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 131072, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 0]},
            {"k": "lit", "v": [1, 1, 131072, 64]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 131072, 64],
            },
        ],
    },
    {
        "id": "11_131072x128_bf8_int-dram",
        "op": "ttnn.slice",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 131072, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [0, 0, 0, 64]},
            {"k": "lit", "v": [1, 1, 131072, 128]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 131072, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_slice(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
