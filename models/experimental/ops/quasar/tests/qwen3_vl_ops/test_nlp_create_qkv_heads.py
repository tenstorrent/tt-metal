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
Per-op test: ``ttnn.experimental.nlp_create_qkv_heads`` — every distinct call the model made, as captured.

Captured 144 call(s) to this op; 2 distinct signature(s) covering 144 of them.

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

_OP = ttnn.experimental.nlp_create_qkv_heads

CASES = [
    {
        "id": "00_12288x3072_bf16_int-dram",
        "op": "ttnn.experimental.nlp_create_qkv_heads",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 12288, 3072],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "num_heads": {"k": "lit", "v": 16},
            "num_kv_heads": {"k": "lit", "v": 16},
            "transpose_k_heads": {"k": "lit", "v": False},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 16, 12288, 64],
            },
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 16, 12288, 64],
            },
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 16, 12288, 64],
            },
        ],
    },
    {
        "id": "01_4096x6144_bf16_int-dram",
        "op": "ttnn.experimental.nlp_create_qkv_heads",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 6144],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "num_heads": {"k": "lit", "v": 32},
            "num_kv_heads": {"k": "lit", "v": 8},
            "transpose_k_heads": {"k": "lit", "v": False},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 4096, 128],
            },
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 8, 4096, 128],
            },
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 8, 4096, 128],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_nlp_create_qkv_heads(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
