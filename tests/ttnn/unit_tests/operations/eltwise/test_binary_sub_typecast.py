# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.eltwise.test_binary_bcast import rand_bf16_gen


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_dtype, output_dtype,",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        ([1, 1, 32, 32], [1, 1, 32, 32]),
        ([3, 1, 64, 64], [3, 1, 64, 64]),
        ([128, 128], [128, 128]),
        ([5, 2, 96], [5, 2, 96]),
        ([5, 3, 128, 64], [1, 3, 128, 1]),
        ([5, 3, 32, 32], [1, 1, 1, 32]),
        ([5, 1, 1, 128], [5, 1, 1, 1]),
        ([1, 71, 7, 7], [7, 7]),
    ),
)
def test_sub_with_fp_dtypes(device, a_shape, b_shape, input_dtype, output_dtype):
    # x_torch = torch.ones(a_shape, dtype=torch.bfloat16) * 16.76255246
    # y_torch = torch.ones(b_shape, dtype=torch.bfloat16) * 4.257
    x_torch, _ = rand_bf16_gen(a_shape, device, min=-100, max=100)
    y_torch, _ = rand_bf16_gen(b_shape, device, min=-100, max=100)
    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    z_torch = golden_fn(x_torch, y_torch)
    # z_torch = z_torch.to(torch.int32)

    x_tt = ttnn.from_torch(x_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 16.75
    y_tt = ttnn.from_torch(y_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 4.25

    z_tt_sub = ttnn.experimental.sub(x_tt, y_tt, dtype=output_dtype)
    tt_out = ttnn.to_torch(z_tt_sub, dtype=torch.int32)

    # print(x_tt)
    # print(y_tt)
    # print(f"Input dtype: {input_dtype}, Output dtype: {output_dtype}")
    # print("z_torch:", z_torch)
    # out = ttnn.to_torch(z_tt_sub)
    # print("tt_out:", z_tt_sub)
    pcc = 0.999
    if output_dtype == ttnn.bfloat4_b:
        pcc = 0.9

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= pcc
    assert status


# (ttnn.bfloat16, ttnn.bfloat8_b),  16.76255246 (16.75 in tt)  - 4.257(4.25 in tt) = tt 12.5, torch w/o typecast 12.5
# (ttnn.bfloat16, ttnn.bfloat8_b),  -16.76255246  - 4.257 = tt -21.0, torch w/o typecast -21.0
# (ttnn.bfloat16, ttnn.bfloat4_b), 16.76255246  - 4.257 = tt 12, torch w/o typecast 12.5
# (ttnn.bfloat16, ttnn.bfloat4_b), -16.76255246  - 4.257 = tt -20.0, torch w/o typecast -21.0


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_dtype, output_dtype,",
    [
        (ttnn.bfloat16, ttnn.uint16),  # only +ve output values work w typecast, -ve output values become 0s
        # (ttnn.bfloat16, ttnn.int32),  # both +ve and -ve output values s=typecasted
        (ttnn.bfloat16, ttnn.uint32),  # only +ve output values work w typecast, -ve output values become 0s
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        ([1, 1, 32, 32], [1, 1, 32, 32]),
        ([3, 1, 64, 64], [3, 1, 64, 64]),
        ([128, 128], [128, 128]),
        ([5, 2, 96], [5, 2, 96]),
        ([5, 3, 128, 64], [1, 3, 128, 1]),
        ([5, 3, 32, 32], [1, 1, 1, 32]),
        ([5, 1, 1, 128], [5, 1, 1, 1]),
        ([1, 71, 7, 7], [7, 7]),
    ),
)
def test_sub_with_uint_dtypes(device, a_shape, b_shape, input_dtype, output_dtype):
    # x_torch = torch.ones(a_shape, dtype=torch.bfloat16) * 16.76255246
    # y_torch = torch.ones(b_shape, dtype=torch.bfloat16) * 4.257
    x_torch, _ = rand_bf16_gen(a_shape, device, min=50, max=100)
    y_torch, _ = rand_bf16_gen(b_shape, device, min=10, max=30)
    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    z_torch = golden_fn(x_torch, y_torch)
    z_torch = z_torch.to(torch.int32)

    x_tt = ttnn.from_torch(x_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 16.75
    y_tt = ttnn.from_torch(y_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 4.25

    z_tt_sub = ttnn.experimental.sub(x_tt, y_tt, dtype=output_dtype)
    tt_out = ttnn.to_torch(z_tt_sub, dtype=torch.int32)

    # print(f"Input dtype: {input_dtype}, Output dtype: {output_dtype}")
    # print("z_torch:", z_torch)
    # print("tt_out:", z_tt_sub)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_dtype, output_dtype,",
    [
        (ttnn.bfloat16, ttnn.int32),  # both +ve and -ve output values s=typecasted
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        ([1, 1, 32, 32], [1, 1, 32, 32]),
        ([3, 1, 64, 64], [3, 1, 64, 64]),
        ([128, 128], [128, 128]),
        ([5, 2, 96], [5, 2, 96]),
        ([5, 3, 128, 64], [1, 3, 128, 1]),
        ([5, 3, 32, 32], [1, 1, 1, 32]),
        ([5, 1, 1, 128], [5, 1, 1, 1]),
        ([1, 71, 7, 7], [7, 7]),
    ),
)
def test_sub_with_int_dtypes(device, a_shape, b_shape, input_dtype, output_dtype):
    # x_torch = torch.ones(a_shape, dtype=torch.bfloat16) * 16.76255246
    # y_torch = torch.ones(b_shape, dtype=torch.bfloat16) * 4.257
    x_torch, _ = rand_bf16_gen(a_shape, device, min=-100, max=100)
    y_torch, _ = rand_bf16_gen(b_shape, device, min=-90, max=90)
    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    z_torch = golden_fn(x_torch, y_torch)
    z_torch = z_torch.to(torch.int32)

    x_tt = ttnn.from_torch(x_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 16.75
    y_tt = ttnn.from_torch(y_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 4.25

    z_tt_sub = ttnn.experimental.sub(x_tt, y_tt, dtype=output_dtype)
    tt_out = ttnn.to_torch(z_tt_sub, dtype=torch.int32)

    # print(f"Input dtype: {input_dtype}, Output dtype: {output_dtype}")
    # print("z_torch:", z_torch)
    # print("tt_out:", z_tt_sub)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


# (ttnn.bfloat16, ttnn.uint16) 16.76255246 (16.75 in tt)  - 4 = tt 13, torch w/o typecast 12.7500, w int typecast 12
# (ttnn.bfloat16, ttnn.uint16) -16.76255246  - 4 = tt 0, torch w/o typecast -20.7500
# (ttnn.bfloat16, ttnn.int32), 16.76255246  - 4 = tt 12, torch w typecast 12, w int typecast 12
# (ttnn.bfloat16, ttnn.int32), -16.76255246  - 4 = tt -20, torch w typecast -20
# (ttnn.bfloat16, ttnn.uint32), 16.76255246  - 4 = tt 12, torch w/o typecast 12.7500, w int typecast 12
# (ttnn.bfloat16, ttnn.uint32), -16.76255246  - 4 = tt 0, torch w/o typecast -20.7500
