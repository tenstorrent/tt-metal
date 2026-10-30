# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_paged_scaled_dot_product_attention_decode.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured call decodes one user against a 1024-page x 8-KV-head bfloat16 cache (128 MB)
# on an 8x4 grid. These keep the captured query ([1,1,32,128] height-sharded on one core),
# page size and head_dim, with an 8-page cache. On Quasar a core may hold only one KV head
# (several per core hangs; see the llama small-grid test's split variant), so the caches
# carry as many KV heads as the grid has cores: 1 KV head on 1 core, 2 on 2 cores.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit, mem_arg, cfg

_OP = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode
_NAME = "ttnn.transformer.paged_scaled_dot_product_attention_decode"

CASES = [
    case(
        "small_1kv_1core",
        "ttnn.transformer.paged_scaled_dot_product_attention_decode",
        [t([1, 1, 32, 128], mem=S.sharded("HEIGHT_SHARDED", [32, 128])), t([8, 1, 32, 128]), t([8, 1, 32, 128])],
        {
            "page_table_tensor": t([1, 8], dtype="INT32", layout="ROW_MAJOR"),
            "cur_pos_tensor": t([1], dtype="INT32", layout="ROW_MAJOR"),
            "scale": lit(0.08838834764831845),
            "sliding_window_size": lit(None),
            "program_config": cfg(
                "SDPAProgramConfig",
                compute_with_storage_grid_size=[1, 1],
                sub_core_grids=None,
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
                max_cores_per_head_batch=1,
            ),
            "memory_config": mem_arg(S.dram()),
        },
        [out([1, 1, 32, 128])],
    ),
    case(
        "small_2kv_2core",
        "ttnn.transformer.paged_scaled_dot_product_attention_decode",
        [t([1, 1, 32, 128], mem=S.sharded("HEIGHT_SHARDED", [32, 128])), t([8, 2, 32, 128]), t([8, 2, 32, 128])],
        {
            "page_table_tensor": t([1, 8], dtype="INT32", layout="ROW_MAJOR"),
            "cur_pos_tensor": t([1], dtype="INT32", layout="ROW_MAJOR"),
            "scale": lit(0.08838834764831845),
            "sliding_window_size": lit(None),
            "program_config": cfg(
                "SDPAProgramConfig",
                compute_with_storage_grid_size=[2, 1],
                sub_core_grids=None,
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
                max_cores_per_head_batch=1,
            ),
            "memory_config": mem_arg(S.dram()),
        },
        [out([1, 1, 32, 128])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_paged_scaled_dot_product_attention_decode_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
