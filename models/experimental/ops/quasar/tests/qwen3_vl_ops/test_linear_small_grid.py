# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_linear.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured decode projections use the DRAM-sharded matmul
# (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig) with the weight width-sharded over
# 12 DRAM banks, whose core assignment (get_optimal_dram_bank_to_logical_worker_assignment)
# is only implemented for Wormhole/Blackhole. The prefill projections use a 2D-mcast
# program config on an 8x8 grid or no program config (the bias cases). These keep the
# prefill forms in bfloat16 at 128 rows: 2D mcast on a 2x1 grid, no program config, and
# with a bias. The DRAM-sharded decode form has no small-grid equivalent on Quasar.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.linear`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit, dtype, mem_arg, cfg

_OP = ttnn.linear
_NAME = "ttnn.linear"

CASES = [
    case(
        "small_mcast2d_128x256x512",
        _NAME,
        [t([1, 1, 128, 256]), t([1, 1, 256, 512])],
        {
            "dtype": dtype("BFLOAT16"),
            "core_grid": lit(None),
            "program_config": cfg(
                "MatmulMultiCoreReuseMultiCastProgramConfig",
                compute_with_storage_grid_size=[2, 1],
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=4,
                out_block_w=8,
                per_core_M=4,
                per_core_N=8,
                transpose_mcast=0,
                fused_activation=None,
                fuse_batch=0,
                allowed_worker_cores=None,
            ),
            "memory_config": mem_arg(S.dram()),
        },
        [out([1, 1, 128, 512])],
    ),
    case(
        "small_noconfig_128x256x512",
        _NAME,
        [t([1, 1, 128, 256]), t([1, 1, 256, 512])],
        {"dtype": dtype("BFLOAT16"), "memory_config": mem_arg(S.dram())},
        [out([1, 1, 128, 512])],
    ),
    case(
        "small_bias_128x256x512",
        _NAME,
        [t([1, 1, 128, 256]), t([256, 512])],
        {"bias": t([512]), "memory_config": mem_arg(S.dram())},
        [out([1, 1, 128, 512])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_linear_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
