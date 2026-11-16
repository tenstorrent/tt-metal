# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_scaled_dot_product_attention.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured calls are prefill attention at 4096-12288 tokens on an 8x4 SDPAProgramConfig
# grid. These keep the non-causal vision form (head_dim 64, scale 1/8) and the causal GQA
# text form (head_dim 128) at 128 query rows on a 2x1 grid, in bfloat16. Prefill SDPA
# spreads its query chunks over however many cores the config names.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.transformer.scaled_dot_product_attention`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit, cfg

_OP = ttnn.experimental.quasar.transformer.scaled_dot_product_attention
_NAME = "ttnn.transformer.scaled_dot_product_attention"

CASES = [
    case(
        "small_mha_1x128x64",
        "ttnn.transformer.scaled_dot_product_attention",
        [t([1, 1, 128, 64]), t([1, 1, 128, 64]), t([1, 1, 128, 64])],
        {
            "is_causal": lit(False),
            "scale": lit(0.125),
            "program_config": cfg(
                "SDPAProgramConfig",
                compute_with_storage_grid_size=[2, 1],
                sub_core_grids=None,
                q_chunk_size=64,
                k_chunk_size=64,
                exp_approx_mode=False,
                max_cores_per_head_batch=16,
            ),
        },
        [out([1, 1, 128, 64])],
    ),
    case(
        "small_gqa_causal_2x64x128",
        "ttnn.transformer.scaled_dot_product_attention",
        [t([1, 2, 64, 128]), t([1, 1, 64, 128]), t([1, 1, 64, 128])],
        {
            "is_causal": lit(True),
            "scale": lit(0.08838834764831845),
            "program_config": cfg(
                "SDPAProgramConfig",
                compute_with_storage_grid_size=[2, 1],
                sub_core_grids=None,
                q_chunk_size=32,
                k_chunk_size=32,
                exp_approx_mode=False,
                max_cores_per_head_batch=16,
            ),
        },
        [out([1, 2, 64, 128])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_scaled_dot_product_attention_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
