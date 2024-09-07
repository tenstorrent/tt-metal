# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from loguru import logger


import ttnn
import pytest

from models.utility_functions import (
    torch_to_tt_tensor,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    is_wormhole_b0,
    is_blackhole,
)
from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from models.experimental.stable_diffusion.tt.downblock_2d import TtDownBlock2D


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_run_downblock_real_input_inference(device, model_location_generator):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_downblock = pipe.unet.down_blocks[3]

    # synthesize the input
    base_address = "down_blocks.3"

    dir_path = model_location_generator("tensor_files", model_subdir="StableDiffusion")
    attr_path = f"{dir_path}/DownBlock2D_inp__attr.pt"
    emb_path = f"{dir_path}/DownBlock2D_inp__emb.pt"
    sample_path = f"{dir_path}/DownBlock2D_inp__sample.pt"

    map_location = torch.device("cpu")
    sample = torch.load(sample_path, map_location=map_location)
    emb = torch.load(emb_path, map_location=map_location)

    kwargs = torch.load(attr_path)

    # execute torch
    torch_output, torch_output_states = unet_downblock(sample, emb)

    # setup tt models
    tt_sample = torch_to_tt_tensor_rm(sample, device)
    tt_emb = torch_to_tt_tensor_rm(emb, device)

    tt_downblock = TtDownBlock2D(**kwargs, state_dict=state_dict, base_address=base_address)

    tt_out, tt_output_states = tt_downblock(tt_sample, tt_emb)
    ttnn.synchronize_device(device)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


@pytest.mark.skip("Test not run")
def test_run_downblock_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_downblock = pipe.unet.down_blocks[0]
    unet_resnet_downblock_module_list = unet_downblock.resnets

    # synthesize the input
    base_address = "down_blocks.0"
    in_channels = unet_resnet_downblock_module_list[0].conv1.in_channels
    out_channels = unet_resnet_downblock_module_list[0].conv2.in_channels
    temb_channels = 1280
    eps = 1e-05
    resnet_groups = 32
    input_shape = [1, in_channels, 32, 32]
    input = torch.randn(input_shape, dtype=torch.float32)

    temb_shape = [out_channels, out_channels]
    temb = torch.randn(temb_shape, dtype=torch.float32)

    unet_out = unet_resnet_downblock_module_list[0](input, None)
    unet_out = unet_resnet_downblock_module_list[1](unet_out, None)
    # execute torch
    torch_output = unet_downblock.downsamplers[0](unet_out)

    # setup tt models
    tt_input = torch_to_tt_tensor(input, device)
    tt_downblock = TtDownBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        dropout=0.0,
        num_layers=2,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=resnet_groups,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        state_dict=state_dict,
        base_address=base_address,
    )

    tt_out = tt_downblock(tt_input, None)[0]
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
