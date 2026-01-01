# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_nlp_create_qkv_heads.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured calls split prefill QKV at 4096-12288 tokens: the vision tower's MHA
# (16 heads x 64) and the text model's GQA (32 query / 8 KV heads x 128). These keep both
# head geometries at 128 tokens.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.experimental.nlp_create_qkv_heads`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit, mem_arg

_OP = ttnn.experimental.nlp_create_qkv_heads
_NAME = "ttnn.experimental.nlp_create_qkv_heads"

CASES = [
    case(
        "small_mha16x64_128",
        _NAME,
        [t([1, 1, 128, 3072])],
        {
            "num_heads": lit(16),
            "num_kv_heads": lit(16),
            "transpose_k_heads": lit(False),
            "memory_config": mem_arg(S.dram()),
        },
        [out([1, 16, 128, 64]), out([1, 16, 128, 64]), out([1, 16, 128, 64])],
    ),
    case(
        "small_gqa32_8x128_128",
        _NAME,
        [t([1, 1, 128, 6144])],
        {
            "num_heads": lit(32),
            "num_kv_heads": lit(8),
            "transpose_k_heads": lit(False),
            "memory_config": mem_arg(S.dram()),
        },
        [out([1, 32, 128, 128]), out([1, 8, 128, 128]), out([1, 8, 128, 128])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_nlp_create_qkv_heads_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
