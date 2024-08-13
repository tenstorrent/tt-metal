# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from functools import partial
import ttnn.deprecated as ttl


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
        ttnn.experimental.tensor.Layout.ROW_MAJOR,
        ttnn.experimental.tensor.Layout.TILE,
    ),
)
def test_min_max_for_dim_hw(device, use_program_cache, shape_dim, kind, layout):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    x = 1.0 + torch.rand(input_shape).bfloat16()

    if kind == "max":
        value = x.max()
    elif kind == "min":
        if N * C % 32 != 0:
            pytest.skip("global min with Tensor dimension N*C not multiple of 32 is not supported at this time.")
        value = x.min()
    elif kind == "mean":
        value = x.mean()
    else:
        raise AttributeError()

    # print(f"x.max/min = {value}")

    dev_x = ttnn.experimental.tensor.Tensor(x, ttnn.experimental.tensor.DataType.BFLOAT16).to(layout).to(device)
    if kind == "max":
        tt_npu = ttnn.experimental.tensor.global_max(dev_x)
    elif kind == "min":
        tt_npu = ttnn.experimental.tensor.global_min(dev_x)
    else:
        assert kind == "mean"
        tt_npu = ttnn.experimental.tensor.global_mean(dev_x)

    tt_dev = tt_npu.cpu().to(ttnn.experimental.tensor.Layout.ROW_MAJOR).to_torch()

    comparison_fn = torch.equal
    if kind == "mean":
        comparison_fn = partial(torch.isclose, atol=1e-1, rtol=1e-2)
    assert comparison_fn(tt_dev[0, 0, 0, 0], torch.Tensor([value]).bfloat16()[0])
