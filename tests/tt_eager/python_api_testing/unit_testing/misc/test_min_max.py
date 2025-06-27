# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from functools import partial
import ttnn


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3),
        ((1, 2, 32, 32), 3),
        ((2, 3, 32, 32), 3),
        ((1, 1, 32, 32), 2),
        ((1, 2, 32, 32), 2),
        ((2, 3, 32, 32), 2),
        ((32, 32, 32, 32), 1),
        ((32, 32, 32, 32), 0),
        ((32, 32, 32, 32), 2),
        ((32, 32, 32, 32), 3),
    ),
)
@pytest.mark.parametrize(
    "kind",
    (
        "min",
        "max",
        "mean",
    ),  # single tile
)
@pytest.mark.parametrize(
    "layout",
    (
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ),
)
def test_min_max_for_dim_hw(device, shape_dim, kind, layout):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    torch_input = 1.0 + torch.rand(input_shape).bfloat16()

    if kind == "max":
        torch_output = torch_input.max()
    elif kind == "min":
        torch_output = torch_input.min()
    elif kind == "mean":
        torch_output = torch_input.mean()
    else:
        raise AttributeError()

    # print(f"x.max/min = {value}")

    tt_input = ttnn.Tensor(torch_input, ttnn.bfloat16).to(layout).to(device)
    if kind == "max":
        tt_npu = ttnn.max(tt_input)
    elif kind == "min":
        tt_npu = ttnn.min(tt_input)
    else:
        assert kind == "mean"
        tt_npu = ttnn.mean(tt_input)

    tt_output = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comparison_fn = torch.equal
    if kind == "mean":
        comparison_fn = partial(torch.isclose, atol=1e-1, rtol=1e-2)
    assert comparison_fn(tt_output, torch_output)
