# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops

import ttnn

from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape, num_features",
    ((torch.Size([1, 2, 3, 4]), 2), (torch.Size([2, 6, 12, 6]), 6)),
)
@pytest.mark.parametrize("momentum", (0.1, 0.2))
@pytest.mark.parametrize("eps, affine, track_running_stats", ((1e-5, True, True),))
@pytest.mark.parametrize("on_device", (False, True))
def test_BatchNorm_fallback(
    input_shape,
    num_features,
    eps,
    momentum,
    affine,
    track_running_stats,
    on_device,
    device,
):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    w = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    b = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    r_m = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    r_v = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    n_b_t = torch.randint(0, 200, [1, 1, 1, 1]).bfloat16().float()
    pt_nn = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    pt_nn.weight = torch.nn.Parameter(w.reshape([num_features]))
    pt_nn.bias = torch.nn.Parameter(b.reshape([num_features]))
    pt_nn.running_mean = r_m.reshape([num_features])
    pt_nn.running_var = r_v.reshape([num_features])
    pt_nn.n_b_t = torch.tensor(n_b_t.item())
    pt_out = pt_nn(x)

    # Test on host RM
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        t0 = t0.to(device)

    w0 = ttnn.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        w0 = w0.to(device)

    b0 = ttnn.Tensor(
        b.reshape(-1).tolist(),
        b.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        b0 = b0.to(device)

    r_m0 = ttnn.Tensor(
        r_m.reshape(-1).tolist(),
        r_m.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        r_m0 = r_m0.to(device)

    r_v0 = ttnn.Tensor(
        r_v.reshape(-1).tolist(),
        r_v.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        r_v0 = r_v0.to(device)

    # Scaler must remain on host
    n_b_t0 = ttnn.Tensor(
        n_b_t.reshape(-1).tolist(),
        n_b_t.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_nn = fallback_ops.BatchNorm2d(
        w0,
        b0,
        r_m0,
        r_v0,
        n_b_t0,
        num_features,
        eps,
        momentum,
        affine,
        track_running_stats,
    )
    t1 = tt_nn(t0)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
