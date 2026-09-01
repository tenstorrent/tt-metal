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
Per-op test: ``ttnn.transformer.scaled_dot_product_attention`` — every distinct call the model made, as captured.

Captured 144 call(s) to this op; 2 distinct signature(s) covering 144 of them.

Fidelity notes for this op:
  * compute_kernel_config is recorded only as an object address in the capture, so it is dropped and the op's default is used (can shift PCC, not shapes)

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

_OP = ttnn.transformer.scaled_dot_product_attention

CASES = [
    {
        "id": "00_16x12288x64_bf8_int-dram",
        "op": "ttnn.transformer.scaled_dot_product_attention",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 16, 12288, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 16, 12288, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 16, 12288, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "is_causal": {"k": "lit", "v": False},
            "scale": {"k": "lit", "v": 0.125},
            "program_config": {
                "kind": "SDPAProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "sub_core_grids": None,
                    "q_chunk_size": 256,
                    "k_chunk_size": 256,
                    "exp_approx_mode": False,
                    "max_cores_per_head_batch": 16,
                },
                "k": "cfg",
            },
        },
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
        "id": "01_32x4096x128_bf8_int-dram",
        "op": "ttnn.transformer.scaled_dot_product_attention",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 4096, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 8, 4096, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 8, 4096, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "is_causal": {"k": "lit", "v": True},
            "sliding_window_size": {"k": "lit", "v": None},
            "scale": {"k": "lit", "v": 0.08838834764831845},
            "program_config": {
                "kind": "SDPAProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "sub_core_grids": None,
                    "q_chunk_size": 256,
                    "k_chunk_size": 256,
                    "exp_approx_mode": False,
                    "max_cores_per_head_batch": 16,
                },
                "k": "cfg",
            },
        },
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
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_scaled_dot_product_attention(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
