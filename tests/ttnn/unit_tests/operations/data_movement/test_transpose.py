# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc


@pytest.mark.parametrize("shape", [[2, 3, 64, 96], [1, 2, 32, 64]])
@pytest.mark.parametrize("dims", [(1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (0, 1)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_transpose_block_float_tile(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_input = ttnn.to_torch(input_tensor)

    output_tensor = ttnn.transpose(input_tensor, dims[0], dims[1])
    assert output_tensor.dtype == dtype
    output = ttnn.to_torch(output_tensor)

    # Transposing the last dim moves values between shared-exponent blocks, so those results are requantized
    # and only match the transposed quantized input within the format's precision. The other pairs keep every
    # block intact and must be exact.
    expected = quantized_input.transpose(dims[0], dims[1]).contiguous()

    assert list(output.shape) == list(expected.shape)
    if 3 in dims:
        assert_with_pcc(expected, output, 0.99)
    else:
        assert_equal(expected, output)
