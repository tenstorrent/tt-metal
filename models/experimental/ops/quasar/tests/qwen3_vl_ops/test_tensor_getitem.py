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
Per-op test: ``ttnn.Tensor.__getitem__`` — every distinct call the model made, as captured.

Captured 978 call(s) to this op; 7 distinct signature(s) covering 978 of them.

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

_OP = ttnn.Tensor.__getitem__

CASES = [
    {
        "id": "00_1x128_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 800,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, 1, None], [None, None, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 128],
            },
        ],
    },
    {
        "id": "01_8x4096x128_bf8_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 144,
        "args": [
            {
                "k": "t",
                "shape": [1, 8, 4096, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, None, None], [None, 2784, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 8, 2784, 128],
            },
        ],
    },
    {
        "id": "02_12288x1024_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 12,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, None, None], [None, 11008, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 11008, 1024],
            },
        ],
    },
    {
        "id": "03_2752x2560_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 12,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2752, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [0, 1, None], [None, None, None], [None, 2560, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2752, 2560],
            },
        ],
    },
    {
        "id": "04_4096x128_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 4,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[0, 1, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 128],
            },
        ],
    },
    {
        "id": "05_4096x128_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 4,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, None, None], [None, None, None], [0, 4096, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 128],
            },
        ],
    },
    {
        "id": "06_2766x2560_bf16_int-dram",
        "op": "ttnn.Tensor.__getitem__",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [2766, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {"k": "slices", "v": [[None, 2766, None], [None, None, None]]},
        ],
        "kwargs": {},
        "outs": [
            None,
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_tensor_getitem(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
