# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger

from models.utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from models.utility_functions import comp_pcc, comp_allclose_and_pcc

import ttnn
from models.experimental.stable_diffusion.tt.feedforward import TtFeedForward


def test_feedforward_inference(device):
    # synthesize the input
    dim = 1280
    dropout = 0
    act = "geglu"
    final_dropout = False
    input_shape = [1, 2, 64, 1280]
    input = torch.randn(input_shape) * 0.01

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    ff = pipe.unet.mid_block.attentions[0].transformer_blocks[0].ff
    torch_output = ff(input)

    # setup tt model
    tt_ff = TtFeedForward(
        dim=dim,
        dropout=dropout,
        activation_fn=act,
        final_dropout=False,
        state_dict=state_dict,
        device=device,
    )
    ttnn.synchronize_device(device)
    tt_input = torch_to_tt_tensor(input, device)
    tt_output = tt_ff(tt_input)
    tt_output = tt_to_torch_tensor(tt_output)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
