# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [[1, 1, 18, 13]])
@pytest.mark.parametrize("padshape", [[1, 1, TILE_HEIGHT, TILE_WIDTH]])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_pad_op(device, in_dtype, shape, padshape, use_multicore):
    torch_input = torch.randn(shape, dtype=torch.bfloat16).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt = ttnn.pad(ttnn_input, padshape, [0, 0, 0, 0], value=0, use_multicore=use_multicore)
    output_tt = ttnn.to_torch(output_tt)
    assert output_tt.shape == torch.Size(padshape)

    shape_diff = list(map(lambda x, y: x - y, padshape, shape))
    output_torch = torch.nn.functional.pad(torch_input, [0, shape_diff[-1], 0, shape_diff[-2]], value=0)
    passing = assert_with_pcc(output_tt, output_torch, 0.9999)
    assert passing
