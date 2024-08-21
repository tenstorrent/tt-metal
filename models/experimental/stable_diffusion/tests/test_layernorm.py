# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2023, TensTorrent, Inc.
# All rights reserved.

# LayerNorm runs with tensor sizes for Stable Diffusion:
# LayerNorm       [1, 2, 1024, 320]       weights = bias =  [320] normalized_shape = 320 / norm_elementwise = True        stable diffusion
# LayerNorm        [1, 2, 64, 1280]       weights = bias =  [1280]        normalized_shape = 1280 / norm_elementwise = True       stable diffusion
# LayerNorm       [1, 2, 256, 640]        weights = bias =  [640] normalized_shape = 640 / norm_elementwise = True        stable diffusion


import torch
from loguru import logger

from models.utility_functions import torch_to_tt_tensor, tt_to_torch_tensor

from models.utility_functions import comp_pcc, comp_allclose_and_pcc

import ttnn

import pytest


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 2, 1024, 320],
        [1, 2, 64, 1280],
        [1, 2, 256, 640],
        [1, 1, 32, 1024],
    ],
)
@pytest.mark.parametrize(
    "normalized_shape_hint",
    [
        (-1,),
    ],
)
@torch.no_grad()
def test_layer_norm(device, input_shape, normalized_shape_hint):
    pcc = 0.99

    N, C, H, W = input_shape

    x = torch.rand((N, C, H, W))
    eps = 1e-3

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        input_shape,  # [N,C,H,W],
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if H % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)

    xt = xt.to(device)
    normalized_shape = list(map(input_shape.__getitem__, normalized_shape_hint))
    golden = torch.nn.functional.layer_norm(x, normalized_shape=normalized_shape, eps=eps)

    xtt_data = ttnn.layer_norm(xt, epsilon=eps).cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    tt_got_back_rm = xtt_data.to_torch()

    torch_output = golden
    tt_output = tt_got_back_rm

    passing = comp_pcc(torch_output, tt_output, pcc=pcc)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output, pcc=pcc))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
