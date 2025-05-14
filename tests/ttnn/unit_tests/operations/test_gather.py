# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
        ([4, 4], [4, 4], 1),  # xfail case
        ([128, 64], [128, 32], 1),  # xfail case
        ([16, 16, 16], [16, 16, 16], 0),  # xfail case
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),  # xfail case
        ([64, 128, 256], [64, 128, 128], 1),  # xfail case
        ([256, 2, 32], [160, 2, 32], 1),  # xfail case
        ([2, 256, 2, 32], [2, 128, 2, 32], 1),  # xfail case
        ([2, 32, 96], [2, 32, 32], 1),  # xfail case
        ([128, 128], [128, 64], 1),  # xfail case
    ],
)
def test_gather_general(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    if dim != -1:
        pytest.xfail(
            reason="uint32 types support does not exist in ttnn.transpose yet which is used in ttnn.gather. See issue: https://github.com/tenstorrent/tt-metal/issues/18057"
        )

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.experimental.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_with_pcc(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
    ],
)
def test_gather_preallocated_output(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)
    output = torch.zeros_like(index, dtype=torch_dtype)

    torch_gather = torch.gather(input, dim, index, out=output)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    ttnn_output = ttnn.from_torch(output, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    ttnn.experimental.gather(ttnn_input, dim, index=ttnn_index, out=ttnn_output)

    assert ttnn_output.shape == index.shape

    assert_with_pcc(torch_gather, ttnn.to_torch(ttnn_output))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1),  # 16 cores
        ([1, 1, 2048, 64], [1, 1, 2048, 32], -1),  # 64 cores
        ([1, 1, 2240, 64], [1, 1, 2240, 32], -1),  # 70 cores
    ],
)
def test_gather_multicore_cases(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.experimental.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_with_pcc(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1, torch.float32, ttnn.float32, ttnn.uint16),
        ([128, 64], [128, 32], 1, torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        ([2, 32, 96], [2, 32, 32], -1, torch.float32, ttnn.float32, ttnn.uint32),
    ],
)
def test_gather_datatype_cases(
    input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype, device
):
    torch.manual_seed(0)

    input = torch.randn(input_shape, dtype=torch_input_datatype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn_input_datatype, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn_index_datatype, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.experimental.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_with_pcc(torch_gather, ttnn.to_torch(ttnn_gather))
