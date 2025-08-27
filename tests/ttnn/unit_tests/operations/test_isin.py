# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose


def select_torch_dtype(ttnn_dtype):
    if ttnn_dtype is ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype is ttnn.float32:
        return torch.float32
    if ttnn_dtype is ttnn.uint8:
        return torch.uint8
    if ttnn_dtype is ttnn.uint16:
        return torch.int64
    if ttnn_dtype is ttnn.int32:
        return torch.int64
    if ttnn_dtype is ttnn.uint32:
        return (
            torch.int64
        )  # !!! there is a strict requirement for the index tensor in Torch to be int64, and there is no int64 in ttnn


@pytest.mark.parametrize(
    "elements, test_elements, dtype",
    [
        (
            [i for i in range(200)],
            [2, 3, 1, 500],
            ttnn.int32,
        ),
        (
            [i for i in range(200)],
            [-100, 200, 300],
            ttnn.int32,
        ),
        (
            [i for i in range(10, 200)],
            [11, 2, 3, 24, 20, 10, 200, 199],
            ttnn.uint16,
        ),
        (
            [[i for i in range(10, 20)] for _ in range(5)],
            [11, 2, 3, 24, 20, 10, 200, 199],
            ttnn.uint32,
        ),
        # (
        #     [[[i ^ j ^ k for i in range(0, 10)] for j in range(0, 10)] for k in range(0, 10)],
        #     [28 * i for i in range(0, 20)],
        #     ttnn.int32
        # )
    ],
)
def test_isin_normal(elements, test_elements, dtype, device):
    torch_dtype = select_torch_dtype(dtype)
    elements_torch = torch.tensor(elements, dtype=torch_dtype)
    test_elements_torch = torch.tensor(test_elements, dtype=torch_dtype)

    elements_ttnn = ttnn.from_torch(elements_torch, device=device)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device)

    torch_isin_result = torch.isin(elements_torch, test_elements_torch)
    ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn)

    assert torch_isin_result.shape == ttnn_isin_result.shape
    assert torch_isin_result.count_nonzero() == ttnn.to_torch(ttnn_isin_result).count_nonzero()
