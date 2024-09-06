# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger


import ttnn
from models.utility_functions import (
    torch_to_tt_tensor,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    is_wormhole_b0,
    is_blackhole,
)
from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from models.experimental.stable_diffusion.tt.upblock_2d import TtUpBlock2D
import pytest


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_run_upblock_real_input_inference(device, model_location_generator):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]

    # synthesize the input
    base_address = "up_blocks.0"

    dir_path = model_location_generator("tensor_files", model_subdir="StableDiffusion")

    attr_path = f"{dir_path}/UpBlock2D_inp__attr.pt"
    emb_path = f"{dir_path}/UpBlock2D_inp__emb.pt"
    sample_path = f"{dir_path}/UpBlock2D_inp__sample.pt"
    res_sample_path = f"{dir_path}/UpBlock2D_inp__res_samples.pt"
    upsample_path = f"{dir_path}/UpBlock2D_inp__upsample_size.pt"

    map_location = torch.device("cpu")

    sample = torch.load(sample_path, map_location=map_location)
    emb = torch.load(emb_path, map_location=map_location)
    res_samples = torch.load(res_sample_path, map_location=map_location)
    upsample_size = torch.load(upsample_path, map_location=map_location)

    kwargs = torch.load(attr_path)

    torch_output = unet_upblock(sample, res_samples, emb, upsample_size)

    tt_sample = torch_to_tt_tensor_rm(sample, device)
    tt_emb = torch_to_tt_tensor_rm(emb, device)
    _ttt = torch_to_tt_tensor_rm
    tt_res_samples = [_ttt(res_samples[i], device) for i in range(len(res_samples))]

    tt_upblock = TtUpBlock2D(**kwargs, state_dict=state_dict, base_address=base_address)
    tt_out = tt_upblock(tt_sample, tt_res_samples, tt_emb, None)
    tt_output = tt_to_torch_tensor(tt_out)

    ttnn.synchronize_device(device)
    passing = comp_pcc(torch_output, tt_output, pcc=0.988)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


@pytest.mark.skip(reason="Test not run")
def test_run_upblock_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]

    # synthesize the input
    base_address = "up_blocks.0"
    in_channels = 1280
    out_channels = 1280
    prev_output_channel = 1280
    temb_channels = None
    eps = 1e-05
    resnet_groups = 32
    input_shape = [2, 1280, 8, 8]
    hidden_state = torch.randn(input_shape, dtype=torch.float32)
    res_hidden_states_tuple = (hidden_state, hidden_state, hidden_state)
    temb_shape = [1, 1, 2, 1280]
    temb = torch.randn(temb_shape)

    # execute pytorch
    torch_output = unet_upblock(hidden_state, res_hidden_states_tuple, None, None)

    # setup tt models
    tt_upblock = TtUpBlock2D(
        in_channels=in_channels,
        prev_output_channel=prev_output_channel,
        out_channels=out_channels,
        temb_channels=temb_channels,
        dropout=0.0,
        num_layers=3,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=resnet_groups,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_upsample=True,
        state_dict=state_dict,
        base_address=base_address,
    )

    tt_out = tt_upblock(hidden_state, res_hidden_states_tuple, None, None)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output, pcc=0.97)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
