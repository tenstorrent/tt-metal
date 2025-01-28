# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_dtype, output_dtype,",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [3, 1, 64, 64],
        [128, 128],
        [5, 2, 96],
    ],
)
def test_sub_with_fp_dtypes(device, shape, input_dtype, output_dtype):
    x_torch = torch.ones([1, 1, 32, 32], dtype=torch.bfloat16) * 16.76255246
    y_torch = torch.ones([1, 1, 32, 32], dtype=torch.bfloat16) * 4.257
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

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
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
        (ttnn.bfloat16, ttnn.int32),  # both +ve and -ve output values s=typecasted
        (ttnn.bfloat16, ttnn.uint32),  # only +ve output values work w typecast, -ve output values become 0s
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [3, 1, 64, 64],
        [128, 128],
        [5, 2, 96],
    ],
)
def test_sub_with_int_dtypes(device, shape, input_dtype, output_dtype):
    x_torch = torch.ones([1, 1, 32, 32], dtype=torch.bfloat16) * 16.76255246
    y_torch = torch.ones([1, 1, 32, 32], dtype=torch.bfloat16) * 4.257
    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    z_torch = golden_fn(x_torch, y_torch)
    z_torch = z_torch.to(torch.int32)

    x_tt = ttnn.from_torch(x_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 16.75
    y_tt = ttnn.from_torch(y_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)  # 4.25

    z_tt_sub = ttnn.experimental.sub(x_tt, y_tt, dtype=output_dtype)
    tt_out = ttnn.to_torch(z_tt_sub, dtype=torch.int32)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


# (ttnn.bfloat16, ttnn.uint16) 16.76255246 (16.75 in tt)  - 4 = tt 13, torch w/o typecast 12.7500, w int typecast 12
# (ttnn.bfloat16, ttnn.uint16) -16.76255246  - 4 = tt 0, torch w/o typecast -20.7500
# (ttnn.bfloat16, ttnn.int32), 16.76255246  - 4 = tt 12, torch w typecast 12, w int typecast 12
# (ttnn.bfloat16, ttnn.int32), -16.76255246  - 4 = tt -20, torch w typecast -20
# (ttnn.bfloat16, ttnn.uint32), 16.76255246  - 4 = tt 12, torch w/o typecast 12.7500, w int typecast 12
# (ttnn.bfloat16, ttnn.uint32), -16.76255246  - 4 = tt 0, torch w/o typecast -20.7500
