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
Per-op test: ``ttnn.reshape`` — every distinct call the model made, as captured.

Captured 2352 call(s) to this op; 14 distinct signature(s) covering 2352 of them.

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

_OP = ttnn.reshape

CASES = [
    {
        "id": "00_32x2880_bf8_int-l1",
        "op": "ttnn.reshape",
        "count": 576,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 2880],
            },
        ],
    },
    {
        "id": "01_32x1x2880_bf8_int-l1",
        "op": "ttnn.reshape",
        "count": 384,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 32, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 1, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 1, 2880],
            },
        ],
    },
    {
        "id": "02_32x1_bf16_int-l1",
        "op": "ttnn.reshape",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [32, 1],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 1)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 1],
            },
        ],
    },
    {
        "id": "03_1x2880_bf8_int-l1",
        "op": "ttnn.reshape",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (-1, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 2880],
            },
        ],
    },
    {
        "id": "04_1x2880_bf8_int-l1",
        "op": "ttnn.reshape",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 1, 2880)},
            {"k": "lit", "v": (1, 1, 32, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 2880],
            },
        ],
    },
    {
        "id": "05_32x2880_bf8_int-l1",
        "op": "ttnn.reshape",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 1, 2880)},
            {"k": "lit", "v": (1, 1, 32, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 2880],
            },
        ],
    },
    {
        "id": "06_32x1x2880_bf8_int-l1",
        "op": "ttnn.reshape",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [32, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 1, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 1, 2880],
            },
        ],
    },
    {
        "id": "07_32x1x4x32x2880_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 96,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 1, 4, 32, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 128, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 128, 2880],
            },
        ],
    },
    {
        "id": "08_32x128x2880_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 96,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 128, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 128, 2880],
            },
        ],
    },
    {
        "id": "09_1x32_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 32)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32],
            },
        ],
    },
    {
        "id": "10_128x2880_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 2880],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (-1, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [128, 2880],
            },
        ],
    },
    {
        "id": "11_128x2880_bf16_int-dram",
        "op": "ttnn.reshape",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 2880],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 4, 32, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 4, 32, 2880],
            },
        ],
    },
    {
        "id": "12_32x128_bf16_int-l1",
        "op": "ttnn.reshape",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [32, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {"k": "lit", "v": (1, 32, 128, 1)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 128, 1],
            },
        ],
    },
    {
        "id": "13_128x2880_bf8_int-dram",
        "op": "ttnn.reshape",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "lit", "v": (1, 1, 128, 2880)},
            {"k": "lit", "v": (1, 1, 128, 2880)},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 128, 2880],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reshape(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
