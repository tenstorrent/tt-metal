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
Per-op test: ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` — every distinct call the model made, as captured.

Captured 308 call(s) to this op; 1 distinct signature(s) covering 308 of them.

Fidelity notes for this op:
  * compute_kernel_config is recorded only as an object address in the capture, so it is dropped and the op's default is used (can shift PCC, not shapes)
  * an integer index tensor is involved; the capture holds no values, so it is filled by graph_case.INDEX_VALUES (page ids, positions, token ids) instead of random data

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

_OP = ttnn.transformer.paged_scaled_dot_product_attention_decode

CASES = [
    {
        "id": "00_32x64_bf16_hs-l1",
        "op": "ttnn.transformer.paged_scaled_dot_product_attention_decode",
        "count": 308,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [128, 8, 32, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [128, 8, 32, 64],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "page_table_tensor": {
                "k": "t",
                "shape": [1, 128],
                "dtype": "INT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "cur_pos_tensor": {
                "k": "t",
                "shape": [1],
                "dtype": "INT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "scale": {"k": "lit", "v": 0.125},
            "sliding_window_size": {"k": "lit", "v": None},
            "program_config": {
                "kind": "SDPAProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "sub_core_grids": None,
                    "q_chunk_size": 0,
                    "k_chunk_size": 0,
                    "exp_approx_mode": True,
                    "max_cores_per_head_batch": 16,
                },
                "k": "cfg",
            },
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 64],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_paged_scaled_dot_product_attention_decode(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
