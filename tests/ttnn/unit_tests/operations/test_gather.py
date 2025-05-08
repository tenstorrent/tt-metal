# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
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
def test_gather_general(input_shape, index_shape, dim, device):
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


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([4, 4], [4, 4], 1),
        ([128, 64], [128, 32], 1),
        ([16, 16, 16], [16, 16, 16], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([64, 128, 256], [64, 128, 128], 1),
        ([256, 2, 32], [160, 2, 32], 1),
        ([2, 256, 2, 32], [2, 128, 2, 32], 1),
        ([2, 32, 96], [2, 32, 32], 1),
        ([128, 128], [128, 64], 1),
    ],
)
def test_gather_not_last_dim(input_shape, index_shape, dim, device):
    """
    The ttnn.transpose does not support uint/int types yet.
    Since this functionality is used inside the gather operation, on index tensor for test purposes for now we do it manually.
    """
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    if dim != -1:
        input = input.transpose(dim, -1)
        index = index.transpose(dim, -1)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.experimental.gather(ttnn_input, -1, index=ttnn_index)

    if dim != -1:
        ttnn_gather = ttnn.from_device(ttnn_gather).to_torch().transpose(dim, -1)
        input = input.transpose(dim, -1)
        index = index.transpose(dim, -1)

    assert ttnn_gather.shape == index.shape
    if dim != -1:
        assert_with_pcc(torch_gather, ttnn_gather)
    else:
        assert_with_pcc(torch_gather, ttnn.to_torch(ttnn_gather))


@skip_for_grayskull()
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


@skip_for_grayskull()
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


@skip_for_grayskull()
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
