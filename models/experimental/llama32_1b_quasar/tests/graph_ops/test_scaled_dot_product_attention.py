# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json --out models/experimental/llama32_1b_quasar/tests/graph_ops
# Source capture: generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.transformer.scaled_dot_product_attention`` — every distinct call the model made, as captured.

Captured 32 call(s) to this op; 1 distinct signature(s) covering 32 of them.

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
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.experimental.quasar.transformer.scaled_dot_product_attention

CASES = [
    {
        "id": "00_32x1024x64_bf16_int-dram",
        "op": "ttnn.transformer.scaled_dot_product_attention",
        "count": 32,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 1024, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 8, 1024, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 8, 1024, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "is_causal": {"k": "lit", "v": True},
            "sliding_window_size": {"k": "lit", "v": None},
            "scale": {"k": "lit", "v": 0.125},
            "program_config": {
                "kind": "SDPAProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "sub_core_grids": None,
                    "q_chunk_size": 64,
                    "k_chunk_size": 64,
                    "exp_approx_mode": False,
                    "max_cores_per_head_batch": 16,
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 1024, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_scaled_dot_product_attention(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
