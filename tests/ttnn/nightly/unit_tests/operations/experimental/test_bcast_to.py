# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Nightly tests for ttnn.experimental.broadcast_to with INT32/UINT32 data types.

These tests cover the MOVD2B path via _llk_math_eltwise_unary_datacopy_ (WH/BH):

  - unpack_to_dest=true + is_32bit_input + BroadcastType::ROW
      → MOVD2B dest_32b_hi=0 then dest_32b_lo=1 in the BroadcastType::ROW path of _llk_math_eltwise_unary_datacopy_
  - unpack_to_dest=true + is_32bit_input + BroadcastType::COL
      → same MOVD2B pattern in the BroadcastType::COL path
  - unpack_to_dest=true + is_32bit_input + BroadcastType::SCALAR
      → same MOVD2B pattern in the BroadcastType::SCALAR path

SubtileBroadcastType is determined by input.logical_shape[-2:] vs output shape:
  ROW:    input_h == 1  and  input_w == output_w
  COL:    input_h == output_h  and  input_w == 1
  SCALAR: input_h == 1  and  input_w == 1  (and output h > 1, w > 1)

"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

torch.manual_seed(0)


def _rand_torch(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**30), 2**30, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


# ──────────────────────────────────────────────────────────────────────────────
# BroadcastType::ROW  (input_h == 1, input_w == output_w)
# llk_math_eltwise_unary_datacopy.h: BroadcastType::ROW branch → MOVD2B dest_32b_lo=1
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([1, 1, 1, 32], [1, 1, 32, 32]),
        ([1, 1, 1, 64], [1, 1, 64, 64]),
        ([1, 3, 1, 32], [1, 3, 32, 32]),
        ([2, 4, 1, 96], [2, 4, 32, 96]),
    ],
)
def test_bcast_to_row_int(device, dtype, input_shape, output_shape):
    """
    ROW broadcast: input_h=1, input_w==output_w.
    Exercises _llk_math_eltwise_unary_datacopy_ BroadcastType::ROW branch (MOVD2B dest_32b_lo=1).
    """
    torch_input = _rand_torch(dtype, input_shape)
    torch_output = torch_input.expand(output_shape)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    tt_output = ttnn.experimental.broadcast_to(tt_input, ttnn.Shape(output_shape))
    assert_equal(torch_output, ttnn.to_torch(tt_output))


# ──────────────────────────────────────────────────────────────────────────────
# BroadcastType::COL  (input_h == output_h, input_w == 1)
# llk_math_eltwise_unary_datacopy.h: BroadcastType::COL branch → MOVD2B dest_32b_lo=1
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([1, 1, 32, 1], [1, 1, 32, 32]),
        ([1, 1, 64, 1], [1, 1, 64, 64]),
        ([1, 3, 32, 1], [1, 3, 32, 32]),
        ([2, 4, 32, 1], [2, 4, 32, 96]),
    ],
)
def test_bcast_to_col_int(device, dtype, input_shape, output_shape):
    """
    COL broadcast: input_w=1, input_h==output_h.
    Exercises _llk_math_eltwise_unary_datacopy_ BroadcastType::COL branch (MOVD2B dest_32b_lo=1).
    """
    torch_input = _rand_torch(dtype, input_shape)
    torch_output = torch_input.expand(output_shape)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    tt_output = ttnn.experimental.broadcast_to(tt_input, ttnn.Shape(output_shape))
    assert_equal(torch_output, ttnn.to_torch(tt_output))


# ──────────────────────────────────────────────────────────────────────────────
# BroadcastType::SCALAR  (input_h == 1 and input_w == 1)
# llk_math_eltwise_unary_datacopy.h: BroadcastType::SCALAR branch → MOVD2B dest_32b_lo=1
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([1, 1, 1, 1], [1, 1, 32, 32]),
        ([1, 1, 1, 1], [1, 1, 64, 64]),
        ([1, 3, 1, 1], [1, 3, 32, 32]),
        ([2, 4, 1, 1], [2, 4, 64, 96]),
    ],
)
def test_bcast_to_scalar_int(device, dtype, input_shape, output_shape):
    """
    SCALAR broadcast: input_h=1 and input_w=1, both H and W are broadcast.
    Exercises _llk_math_eltwise_unary_datacopy_ BroadcastType::SCALAR branch (MOVD2B dest_32b_lo=1).
    """
    torch_input = _rand_torch(dtype, input_shape)
    torch_output = torch_input.expand(output_shape)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    tt_output = ttnn.experimental.broadcast_to(tt_input, ttnn.Shape(output_shape))
    assert_equal(torch_output, ttnn.to_torch(tt_output))
