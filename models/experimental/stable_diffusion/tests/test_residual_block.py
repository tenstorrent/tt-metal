# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from loguru import logger


import ttnn
from models.utility_functions import (
    torch_to_tt_tensor,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from models.experimental.stable_diffusion.tt.residual_block import TtResnetBlock2D


def test_run_resnet_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    ############ start of residual block #############
    in_channels = 1280
    out_channels = 1280
    conv_shortcut = False
    dropout = 0
    temb_channels = 1280
    groups = 32
    groups_out = None
    pre_norm = True
    eps = 1e-05
    non_linearity = "silu"
    time_embedding_norm = "default"
    kernel = None
    output_scale_factor = 1
    use_in_shortcut = False
    up = False
    down = False
    ########## end of residual block #############
    hidden_states_shape = [2, 1280, 8, 8]
    temb_shape = [1, 1, 2, 1280]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    base_address = "mid_block.resnets.0"
    resnet = pipe.unet.mid_block.resnets[0]

    torch_output = resnet(input, temb.squeeze(0).squeeze(0))

    # setup tt model
    tt_resnet = TtResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        groups=groups,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_temb = torch_to_tt_tensor_rm(temb, device, put_on_device=False)

    tt_out = tt_resnet(tt_input, tt_temb)
    ttnn.synchronize_device(device)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
