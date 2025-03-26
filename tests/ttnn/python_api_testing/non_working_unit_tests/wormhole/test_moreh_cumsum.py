# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import disable_persistent_kernel_cache

aten = torch.ops.aten


def run_moreh_cumsum_tests(
    in_data,
    device,
):
    x = torch.tensor(in_data[0]).to(torch.int32)

    try:
        x = x.reshape(len(in_data[0]), 1, 1, 1)
        ref_value = aten.cumsum.default(
            x,
            0,
        )

        tt_x = ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

        tt_result = ttnn.moreh_cumsum(
            tt_x,
            0,
        )
        tt_result = ttnn.to_torch(tt_result)

        passed, message = check_with_pcc(ref_value, tt_result)
        assert passed, f"{message, tt_x}"

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.999)


test_sweep_args = [
    ([3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],),
]


@pytest.mark.parametrize(
    "x",
    (test_sweep_args),
)
def test_moreh_cumsum(x, device):
    disable_persistent_kernel_cache()
    run_moreh_cumsum_tests(x, device)


def run_moreh_cumsum_tests_2(
    in_shape,
    torch_type,
    dtype,
    dim,
    device,
):
    x = (torch.rand(in_shape) * 10).to(torch_type)

    try:
        ref_value = aten.cumsum.default(
            x,
            dim,
        )

        tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        tt_result = ttnn.moreh_cumsum(
            tt_x,
            dim,
        )
        tt_result = ttnn.to_torch(tt_result)

        passed, message = check_with_pcc(ref_value, tt_result)
        assert passed, f"{message, tt_x}"

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.999)


test_sweep_args_2 = [
    (
        (3, 27, 17, 14),
        torch.int32,
        ttnn.uint32,
        1,
    ),
    ((3, 27, 17, 14), torch.int32, ttnn.uint32, 0),
    ((21, 27, 17, 14), torch.int32, ttnn.uint32, 1),
    ((3, 27, 33, 34), torch.int32, ttnn.uint32, 1),
    ((129, 27, 17, 14), torch.float32, ttnn.bfloat16, 1),
    ((129, 27, 17, 14), torch.int32, ttnn.uint32, 1),
]


@pytest.mark.parametrize(
    "shape, torch_type, dtype, dim",
    (test_sweep_args_2),
)
def test_moreh_cumsum_2(shape, torch_type, dtype, dim, device):
    run_moreh_cumsum_tests_2(shape, torch_type, dtype, dim, device)
