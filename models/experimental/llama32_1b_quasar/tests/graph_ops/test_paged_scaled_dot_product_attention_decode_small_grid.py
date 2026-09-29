# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated
# test_paged_scaled_dot_product_attention_decode.py — do NOT regenerate this file
# from a capture.
#
# The captured case pins SDPAProgramConfig.compute_with_storage_grid_size = [8, 8]
# (64 cores). sdpa_decode_program_factory.cpp:194 then FATALs on any device whose
# compute grid has fewer than 64 cores ("Cores available (64) exceeds grid size").
#
# Nothing about this problem actually needs 64 cores: it is a paged decode with
# batch B = page_table.padded_shape[0] = 1 (factory line 124-126), 8 KV heads, non-
# MLA. The only core minimums are `num_cores_available <= device grid` (line 194)
# and `num_cores_available >= B` = >= 1 (line 199); the core-allocation math
# (lines 202-211) scales to any core count (2 cores -> 4 KV heads/core; 32 cores ->
# 4 cores/KV head). So we reuse the captured case verbatim and only shrink the grid.
#
# Each variant SKIPS (rather than FATALs) when the device compute grid is smaller
# than the variant's grid rectangle, so the file is safe to run on any device.
# ---------------------------------------------------------------------------
"""Small-grid variants (2 and 32 cores) of ``paged_scaled_dot_product_attention_decode``."""

import copy

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode

# The captured signature (see the generated test), reused verbatim except for the
# program config's compute grid, which each variant overrides below.
_BASE_CASE = {
    "op": "ttnn.transformer.paged_scaled_dot_product_attention_decode",
    "count": 1,
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
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
        {
            "k": "t",
            "shape": [128, 8, 32, 64],
            "dtype": "BFLOAT16",
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
                "compute_with_storage_grid_size": [8, 8],  # overridden per variant
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
}


def _case_with_grid(case_id, grid_xy):
    case = copy.deepcopy(_BASE_CASE)
    case["id"] = case_id
    case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"] = list(grid_xy)
    return case


CASES = [
    _case_with_grid("2core_2x1_bf16", (2, 1)),
    _case_with_grid("32core_8x4_bf16", (8, 4)),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_paged_scaled_dot_product_attention_decode_small_grid(ttnn_mesh_device, reset_seeds, case):
    grid_x, grid_y = case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"]
    dev = ttnn_mesh_device.compute_with_storage_grid_size()
    if dev.x < grid_x or dev.y < grid_y:
        pytest.skip(f"case needs a {grid_x}x{grid_y} grid rectangle; device compute grid is {dev.x}x{dev.y}")
    G.run_case(_OP, case, ttnn_mesh_device)
