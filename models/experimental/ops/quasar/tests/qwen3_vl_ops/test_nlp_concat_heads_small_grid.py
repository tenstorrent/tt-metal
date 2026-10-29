# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_nlp_concat_heads.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured calls concatenate prefill attention output at 4096-12288 tokens (16 heads x
# 64 and 32 heads x 128, bfloat8_b). These keep both head dims in bfloat16 with 4 heads of
# 32 tokens, which keeps the input at the emulator's 128-row limit (heads x tokens).
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.experimental.nlp_concat_heads`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, mem_arg

_OP = ttnn.experimental.nlp_concat_heads
_NAME = "ttnn.experimental.nlp_concat_heads"

CASES = [
    case("small_4x32x64", _NAME, [t([1, 4, 32, 64])], {"memory_config": mem_arg(S.dram())}, [out([1, 1, 32, 256])]),
    case("small_4x32x128", _NAME, [t([1, 4, 32, 128])], {"memory_config": mem_arg(S.dram())}, [out([1, 1, 32, 512])]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_nlp_concat_heads_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
