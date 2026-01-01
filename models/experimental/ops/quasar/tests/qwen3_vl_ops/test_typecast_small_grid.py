# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_typecast.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# Every captured typecast converts bfloat16 to bfloat8_b, which Quasar does not have. These
# run the same op between the float formats Quasar does support, bfloat16 <-> float32, at
# the emulator's 128-row limit.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.typecast`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, dtype

_OP = ttnn.typecast
_NAME = "ttnn.typecast"

CASES = [
    case(
        "small_bf16_to_fp32_4x32x64",
        _NAME,
        [t([1, 4, 32, 64])],
        {"dtype": dtype("FLOAT32")},
        [out([1, 4, 32, 64], dtype="FLOAT32")],
    ),
    case(
        "small_fp32_to_bf16_1x128x128",
        _NAME,
        [t([1, 1, 128, 128], dtype="FLOAT32")],
        {"dtype": dtype("BFLOAT16")},
        [out([1, 1, 128, 128])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_typecast_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
