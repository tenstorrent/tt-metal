# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_layer_norm.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured norms are the vision LayerNorm at 2752-12288 rows (widths 1024 and 4096),
# interleaved in DRAM with [1, 32, W] weight and bias. These keep both widths and the
# weight/bias layout at 128 and 64 rows; graph_case applies the model's compute config.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.layer_norm`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit

_OP = ttnn.layer_norm
_NAME = "ttnn.layer_norm"

CASES = [
    case(
        "small_128x1024",
        _NAME,
        [t([1, 1, 128, 1024])],
        {"weight": t([1, 32, 1024]), "bias": t([1, 32, 1024]), "epsilon": lit(1e-06)},
        [out([1, 1, 128, 1024])],
    ),
    case(
        "small_64x4096",
        _NAME,
        [t([1, 1, 64, 4096])],
        {"weight": t([1, 32, 4096]), "bias": t([1, 32, 4096]), "epsilon": lit(1e-06)},
        [out([1, 1, 64, 4096])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_layer_norm_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
