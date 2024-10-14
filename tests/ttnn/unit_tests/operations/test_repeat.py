# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from functools import reduce

dtypes = [torch.bfloat16]
shapes = [(1, 2, 4, 4), (1, 1, 1, 1)]
repeat_specs = [(1, 2, 1, 1), (2, 2, 2, 2)]

shape_and_repeat_specs = list(zip(shapes, repeat_specs))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape_and_repeat_spec", shape_and_repeat_specs)
def test_repeat(device, dtype, shape_and_repeat_spec):
    shape, repeat_shape = shape_and_repeat_spec
    if dtype == torch.bfloat16 and shape[-1] < 2 and repeat_shape[-1] < 2:
        pytest.skip("bfloat16 needs 4 byte inner dim on the output.")

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=dtype).reshape(shape)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)
