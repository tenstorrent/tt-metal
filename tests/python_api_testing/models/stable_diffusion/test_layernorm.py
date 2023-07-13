# Copyright (C) 2023, TensTorrent, Inc.
# All rights reserved.

# LayerNorm runs with tensor sizes for Stable Diffusion:
# LayerNorm       [1, 2, 1024, 320]       weights = bias =  [320] normalized_shape = 320 / norm_elementwise = True        stable diffusion
# LayerNorm        [1, 2, 64, 1280]       weights = bias =  [1280]        normalized_shape = 1280 / norm_elementwise = True       stable diffusion
# LayerNorm       [1, 2, 256, 640]        weights = bias =  [640] normalized_shape = 640 / norm_elementwise = True        stable diffusion

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger

from utility_functions_new import (
    torch_to_tt_tensor,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)

import tt_lib as ttl

import torch
import tt_lib

import pytest


@pytest.mark.parametrize(
    "input_shape",
    [[1, 1, 32, 32], [1, 2, 1024, 320], [1, 2, 64, 1280], [1, 2, 256, 640], [1, 1, 16, 1024]],
)
@pytest.mark.parametrize(
    "normalized_shape_hint",
    [
        (-1,),
    ],
)
@torch.no_grad()
def test_layer_norm(input_shape, normalized_shape_hint):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

    pcc = 0.99

    N, C, H, W = input_shape

    x = torch.rand((N, C, H, W))
    eps = 1e-3

    xt = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            input_shape,  # [N,C,H,W],
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR
        )
    if H % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)

    xt = xt.to(device)
    normalized_shape = list(map(input_shape.__getitem__, normalized_shape_hint))
    golden = torch.nn.functional.layer_norm(
        x, normalized_shape=normalized_shape, eps=eps
    )

    xtt_data = tt_lib.tensor.layernorm(xt, eps).to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    tt_got_back_rm = torch.Tensor(xtt_data).reshape(input_shape)

    torch_output = golden
    tt_output = tt_got_back_rm

    passing = comp_pcc(torch_output, tt_output, pcc=pcc)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output, pcc=pcc))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
