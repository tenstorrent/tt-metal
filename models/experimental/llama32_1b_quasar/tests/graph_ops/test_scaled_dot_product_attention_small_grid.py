# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_scaled_dot_product_attention.py — do NOT
# regenerate this file from a capture.
#
# The captured case pins SDPAProgramConfig.compute_with_storage_grid_size = [8, 8]
# (64 cores). sdpa_program_factory.cpp:379 then FATALs on any device whose compute
# grid has fewer than 64 cores ("num_cores (64) exceeds grid size").
#
# Nothing about this problem needs 64 cores: prefill SDPA distributes the flat
# total_q_chunks = B * NQH * q_num_chunks Q-chunk space evenly across num_cores
# (factory lines 387-402), with the only core constraints being `num_cores <= device
# grid` (line 379) and `num_cores > 0` (line 385).
#
# Three deviations from the capture, required to run (and finish) on Quasar:
#   * K and V were captured as BFLOAT8_B; Quasar has no bf8, so they are bf16 here
#     (mirrors the test_add.py bf8 -> bf16 fix). Q and the output were already bf16.
#   * compute_with_storage_grid_size is overridden per variant (2 and 32 cores).
#   * The captured seq len is 1024 with q/k_chunk 64 -> q_num_chunks 16 -> 512 flat
#     Q-chunks, each over up to 16 causal K-chunks. On the functional simulator (which
#     is ~serial over total work, so a smaller grid does NOT speed it up) that is
#     thousands of flash-attention chunk-pairs and runs for many minutes. We shrink to
#     seq=128 with q/k_chunk=128 -> q_num_chunks=1 -> 32 flat Q-chunks, matching the
#     ops/ seq128 SDPA test that completes in ~55s on the sim. This exercises the same
#     prefill kernels/factory at a sim-tractable size; it is NOT the model's real seq.
#
# Each variant SKIPS (rather than FATALs) when the device compute grid is smaller
# than the variant's grid rectangle, so the file is safe to run on any device.
# ---------------------------------------------------------------------------
"""Small-grid variants (2 and 32 cores) of ``scaled_dot_product_attention`` (prefill)."""

import copy

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.experimental.quasar.transformer.scaled_dot_product_attention

# The captured signature (see the generated test), reused verbatim except that the
# bf8 K/V are bf16 (Quasar has no bf8) and the program config's compute grid is
# overridden per variant below.
_BASE_CASE = {
    "op": "ttnn.transformer.scaled_dot_product_attention",
    "count": 1,
    "args": [
        {
            "k": "t",
            "shape": [1, 32, 128, 64],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
        {
            # Captured as BFLOAT8_B; Quasar has no bf8 -> bf16.
            "k": "t",
            "shape": [1, 8, 128, 64],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
        {
            # Captured as BFLOAT8_B; Quasar has no bf8 -> bf16.
            "k": "t",
            "shape": [1, 8, 128, 64],
            "dtype": "BFLOAT16",
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
                "compute_with_storage_grid_size": [8, 8],  # overridden per variant
                "sub_core_grids": None,
                "q_chunk_size": 128,
                "k_chunk_size": 128,
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
            "shape": [1, 32, 128, 64],
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
def test_scaled_dot_product_attention_small_grid(ttnn_mesh_device, reset_seeds, case):
    grid_x, grid_y = case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"]
    dev = ttnn_mesh_device.compute_with_storage_grid_size()
    if dev.x < grid_x or dev.y < grid_y:
        pytest.skip(f"case needs a {grid_x}x{grid_y} grid rectangle; device compute grid is {dev.x}x{dev.y}")
    G.run_case(_OP, case, ttnn_mesh_device)
