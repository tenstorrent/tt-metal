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
Per-op test: ``ttnn.reshape`` — every distinct call the model made, as captured.

Captured 30641 call(s) to this op; 20 distinct signature(s) covering 30241 of them.

Fidelity notes for this op:
  * an integer index tensor is involved; the capture holds no values, so it is filled by graph_case.INDEX_VALUES (page ids, positions, token ids) instead of random data

NOT generated (arguments not reconstructible from the capture):
  * 400 call(s): argument(s) 1

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

_OP = ttnn.reshape

CASES = [
    {
        "id": "00_32x6144_bf16_int-l1",
        "op": "ttnn.reshape",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 6144],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 1, 6144)},
            {"k": "lit", "v": (1, 1, 32, 6144)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 6144],
            },
        ],
    },
    {
        "id": "01_32x2560_bf16_ws-l1",
        "op": "ttnn.reshape",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 160], "orientation": "ROW_MAJOR"},
                },
            },
            {"k": "lit", "v": (1, 1, 32, 2560)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 1]], "orientation": "ROW_MAJOR", "shape": [32, 160]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "02_1_u32_int-dram",
        "op": "ttnn.reshape",
        "count": 400,
        "args": [
            {
                "k": "t",
                "shape": [1],
                "dtype": "UINT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "UINT32",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1],
            },
        ],
    },
    {
        "id": "03_12288x1024_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 12, 1024, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 12, 1024, 1024],
            },
        ],
    },
    {
        "id": "04_12288x1024_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 6, 2048, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 6, 2048, 1024],
            },
        ],
    },
    {
        "id": "05_4096x2560_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 2, 2048, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 2, 2048, 2560],
            },
        ],
    },
    {
        "id": "06_4096x2560_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 4, 1024, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 4, 1024, 2560],
            },
        ],
    },
    {
        "id": "07_12x1024x1024_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 12, 1024, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 12288, 1024)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 12288, 1024],
            },
        ],
    },
    {
        "id": "08_2x2048x6144_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 2, 2048, 6144],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 1, 4096, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 6144],
            },
        ],
    },
    {
        "id": "09_6x2048x3072_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 6, 2048, 3072],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 1, 12288, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 12288, 3072],
            },
        ],
    },
    {
        "id": "10_12288x1024_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 12, 1024, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 12, 1024, 1024],
            },
        ],
    },
    {
        "id": "11_4096x4096_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 4096],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 4, 1024, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 4, 1024, 4096],
            },
        ],
    },
    {
        "id": "12_12x1024x1024_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 12, 1024, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 1, 12288, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 12288, 1024],
            },
        ],
    },
    {
        "id": "13_16x12288x64_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 16, 12288, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 16, -1, 64]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 16, 12288, 64],
            },
        ],
    },
    {
        "id": "14_32x4096x128_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 4096, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 32, -1, 128]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 4096, 128],
            },
        ],
    },
    {
        "id": "15_4x1024x2560_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 4, 1024, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": [1, 1, 4096, -1]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "16_4x1024x2560_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 4, 1024, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 4096, 2560)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "17_11008x1024_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 12,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 11008, 1024],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, -1, 4096)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2752, 4096],
            },
        ],
    },
    {
        "id": "18_2752x2560_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 12,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2752, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (-1, 2560)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [2752, 2560],
            },
        ],
    },
    {
        "id": "19_2752x4096_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 9,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2752, 4096],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, -1, 4096)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2752, 4096],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reshape(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
