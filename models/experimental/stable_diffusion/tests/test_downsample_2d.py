# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline
from loguru import logger

import ttnn
from models.utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from models.utility_functions import comp_allclose_and_pcc, comp_pcc

from models.experimental.stable_diffusion.tt.downsample_2d import TtDownsample2D


def test_run_downsample2d_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_downblock = pipe.unet.down_blocks[0]
    resnet_downsampler = unet_downblock.downsamplers[0]

    # synthesize the input
    input_shape = [1, 320, 32, 32]
    input = torch.randn(input_shape)
    in_channels = 320
    out_channels = 320

    # excute pytorch
    torch_output = resnet_downsampler(input)

    # setup tt models
    tt_input = torch_to_tt_tensor(input, device)

    tt_down = TtDownsample2D(
        channels=in_channels,
        out_channels=out_channels,
        use_conv=True,
        state_dict=state_dict,
        base_address="down_blocks.0.downsamplers.0",
    )
    tt_out = tt_down(tt_input)
    ttnn.synchronize_device(device)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
