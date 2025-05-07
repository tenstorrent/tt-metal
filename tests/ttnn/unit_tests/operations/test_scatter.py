# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def select_torch_dtype(ttnn_dtype):
    pass


@pytest.mark.parametrize("input_shape", [])
@pytest.mark.parametrize("dim", [])
@pytest.mark.parametrize("index_and_src_shape", [])
@pytest.mark.parametrize("input_dtype", [])
def test_scatter_floating_point(input_shape, dim, index_and_src_shape, input_dtype, device):
    # input_torch = torch.
    torch.manual_seed(22041997)

    torch_dtype = select_torch_dtype(input_dtype)
    ##
    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.Layout.TILE, device=device)

    torch_index = torch.randint(0, index_and_src_shape, dtype=torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.Layout.TILE, device=device)

    torch_src = torch.randn(index_and_src_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=ttnn.Layout.TILE, device=device)

    assert_with_pcc()

    torch_input_preallocated = torch.zeros(torch_input.shape)
    ttnn_input_preallocated = torch.zeros(torch_input.shape)


@pytest.mark.parametrize("input_shape", [])
@pytest.mark.parametrize("dim", [])
@pytest.mark.parametrize("index_and_src_shape", [])
@pytest.mark.parametrize("input_ttnn_dtype", [])
def test_scatter_integer(input_shape, dim, index_and_src_shape, input_ttnn_dtype, device):
    # input_torch = torch.
    torch.manual_seed(22041997)
    #
    torch_dtype = select_torch_dtype(input_ttnn_dtype)
    #
    torch_input = torch.zeros(input_shape, dtype=torch_dtype)
    #
    pass
