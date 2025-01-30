# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from functools import reduce

dtypes = [torch.bfloat16]
# shapes = [(17, 17), (1, 1, 1, 1), (1, 1, 1, 64), (1, 1, 1, 1), (1, 1, 64, 64), (1, 1, 64, 64)]
# broadcast_specs = [
#     (1, 31, 17, 17),
#     (1, 1, 170, 17),
#     # (1, 1, 64 * 160, 64),
#     # (1, 1, 640, 640),
#     # (1, 320, 64, 64),
#     # (320, 1, 64, 64),
# ]
shapes = [
    (1, 1, 1, 1),
    (1, 1, 1, 1),
    (1, 1, 1, 1),
    (1, 1, 270, 270),
    (1, 1, 64, 64),
    (1, 1, 64, 64),
    (1, 1, 64, 64),
    (31, 33, 64, 64),
]
broadcast_specs = [
    (1, 1, 1, 67000),
    (1, 1, 67000, 4),
    (1, 1, 270, 270),
    (1, 1, 270, 270),
    (1, 310, 64, 64),
    (310, 1, 64, 64),
    (1, 310, 64, 64),
    (31, 33, 64, 64),
]

shape_and_broadcast_specs = list(zip(shapes, broadcast_specs))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape_and_broadcast_spec", shape_and_broadcast_specs)
def test_broadcast_to(device, dtype, shape_and_broadcast_spec):
    shape, broadcast_shape = shape_and_broadcast_spec
    if dtype == torch.bfloat16 and shape[-1] < 2 and broadcast_shape[-1] < 2:
        pytest.skip("bfloat16 needs 4 byte inner dim on the output.")

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=dtype).reshape(shape)

    torch_result = torch_input_tensor.broadcast_to(broadcast_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.experimental.broadcast_to(input_tensor, ttnn.Shape(broadcast_shape))
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)
