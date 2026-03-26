# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3),
        ((1, 1, 32, 32), 2),
        ((32, 32, 32, 32), 1),
        ((32, 32, 32, 32), 0),
    ),  # single tile
)
def test_sum_for_dim_hw(device, shape_dim):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    x = 1.0 + torch.arange(0, N * C * H * W).reshape(input_shape).bfloat16()

    torch_output = x.sum(dim=dim, keepdim=True)
    value = torch_output[0, 0, 0, 0]

    dev_x = ttnn.Tensor(x, ttnn.DataType.BFLOAT16).to(ttnn.Layout.TILE).to(device)
    tt_npu = ttnn.sum(dev_x, dim=dim, keepdim=True)
    tt_dev = tt_npu.cpu().to(ttnn.Layout.ROW_MAJOR).to_torch()
    # test for equivalance
    assert_numeric_metrics(
        torch_output,
        tt_dev,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=66846.721,
        frobenius_threshold=0.001,
    )
    assert torch.equal(tt_dev[0, 0, 0, 0], torch.Tensor([value]).bfloat16()[0])


@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 32, 32),
        (1, 1, 32, 32),
        (32, 32, 32, 32),
        (32, 32, 32, 32),
    ),  # single tile
)
def test_sum_global(device, shape):
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    x = 1.0 + torch.ones(input_shape).bfloat16()

    torch_output = x.sum()

    dev_x = ttnn.Tensor(x, ttnn.DataType.BFLOAT16).to(ttnn.Layout.TILE).to(device)
    tt_npu = ttnn.sum(dev_x)
    tt_dev = tt_npu.cpu().to(ttnn.Layout.ROW_MAJOR).to_torch()
    # test for equivalance
    assert_numeric_metrics(
        torch_output,
        tt_dev,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )
