# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_multiply.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured multiplies are the MLP gate (SiLU(a) * b, bfloat8_b output) width-sharded
# over 16 cores and at prefill size, and the KV-cache reset (cache * 0 written in place).
# These keep each form in bfloat16: the SiLU-fused gate width-sharded over 2 cores and
# interleaved in DRAM, and the in-place multiply-by-zero on a small cache.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.multiply`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, lit, dtype, mem_arg, acts

# The base binary_ng op still builds Gen1 DataMovementKernels, which Quasar rejects; its
# ttnn.experimental.quasar port takes the same arguments and runs on every arch. _NAME keeps
# the captured op name, so the golden lookup is unchanged.
_OP = ttnn.experimental.quasar.multiply
_NAME = "ttnn.multiply"

CASES = [
    case(
        "small_silu_ws2_32x256",
        _NAME,
        [
            t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
            t([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
        ],
        {
            "input_tensor_a_activations": acts("SILU"),
            "dtype": dtype("BFLOAT16"),
            "memory_config": mem_arg(S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2)),
        },
        [out([1, 1, 32, 256], mem=S.sharded("WIDTH_SHARDED", [32, 128], cores_x=2))],
    ),
    case(
        "small_silu_int_128x512",
        _NAME,
        [t([1, 1, 128, 512]), t([1, 1, 128, 512])],
        {"input_tensor_a_activations": acts("SILU"), "dtype": dtype("BFLOAT16"), "memory_config": mem_arg(S.dram())},
        [out([1, 1, 128, 512])],
    ),
    case("small_zero_inplace_1x4x32x128", _NAME, [t([1, 4, 32, 128]), lit(0)], {"output_tensor": t([1, 4, 32, 128])}),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_multiply_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
