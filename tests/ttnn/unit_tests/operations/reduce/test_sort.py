# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull()
@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([16, 1], -1, False),
        ([1, 16], -1, False),
        ([3, 3], -1, False),
        ([16, 16, 16], -1, True),
        ([3, 3], 1, False),
    ],
)
def test_sort_output_shape(shape, dim, descending, device):
    torch.manual_seed(2005)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape
