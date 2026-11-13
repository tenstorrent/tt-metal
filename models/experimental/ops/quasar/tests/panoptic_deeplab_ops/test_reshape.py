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
Per-op test: ``ttnn.reshape`` — every distinct call the model made, as captured.

Captured 37 call(s) to this op; 12 distinct signature(s) covering 37 of them.

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

_OP = ttnn.reshape

CASES = [
    {
        "id": "00_2048x256_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 10,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 64, 256)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 256],
            },
        ],
    },
    {
        "id": "01_32768x256_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 5,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 128, 256, 256)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 256],
            },
        ],
    },
    {
        "id": "02_32768x32_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 5,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 32],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 128, 256, 32)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 32],
            },
        ],
    },
    {
        "id": "03_32x64x2048_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 4,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 2048, 2048)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2048, 2048],
            },
        ],
    },
    {
        "id": "04_1x2048_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2048],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 1, 2048)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 2048],
            },
        ],
    },
    {
        "id": "05_32768x128_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32768, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 128, 256, 128)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 128, 256, 128],
            },
        ],
    },
    {
        "id": "06_8192x128_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 64, 128, 128)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 128],
            },
        ],
    },
    {
        "id": "07_8192x256_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 64, 128, 256)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 256],
            },
        ],
    },
    {
        "id": "08_8192x64_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 64, 128, 64)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 64],
            },
        ],
    },
    {
        "id": "09_2048x1024_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 64, 1024)},
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
        "id": "10_2048x2048_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2048, 2048],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 64, 2048)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 64, 2048],
            },
        ],
    },
    {
        "id": "11_8192x512_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8192, 512],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 64, 128, 512)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 64, 128, 512],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reshape(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
