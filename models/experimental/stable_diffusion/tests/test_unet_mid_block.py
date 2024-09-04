# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline

import ttnn
from models.utility_functions import (
    torch_to_tt_tensor,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    skip_for_wormhole_b0,
)
from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from models.experimental.stable_diffusion.tt.unet_2d_blocks import TtUNetMidBlock2DCrossAttn
from loguru import logger
import pytest


@skip_for_wormhole_b0()
@pytest.mark.skip(reason="Test is failing, see issue #7536")
def test_run_unet_mid_block_real_input_inference(device, model_location_generator):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    mid_block = pipe.unet.mid_block

    # synthesize the input
    base_address = "down_blocks.3"

    dir_path = model_location_generator("tensor_files", model_subdir="StableDiffusion")
    attention_mask_path = f"{dir_path}/UNetMidBlock2DCrossAttn_inp__attention_mask.pt"
    emb_path = f"{dir_path}/UNetMidBlock2DCrossAttn_inp__emb.pt"
    sample_path = f"{dir_path}/UNetMidBlock2DCrossAttn_inp__sample.pt"
    attr_path = f"{dir_path}/UNetMidBlock2DCrossAttn_inp__attr.pt"
    encoder_hidden_states_path = f"{dir_path}/UNetMidBlock2DCrossAttn_inp__encoder_hidden_states.pt"
    cross_attention_kwargs_path = f"{dir_path}/UNetMidBlock2DCrossAttn_inp__cross_attention_kwargs.pt"

    map_location = torch.device("cpu")
    sample = torch.load(sample_path, map_location=map_location)
    emb = torch.load(emb_path, map_location=map_location)
    attention_mask = torch.load(attention_mask_path, map_location=map_location)
    encoder_hidden_states = torch.load(encoder_hidden_states_path, map_location=map_location)
    cross_attention_kwargs = torch.load(cross_attention_kwargs_path, map_location=map_location)

    kwargs = torch.load(attr_path)

    torch_output = mid_block(
        sample,
        emb,  # .squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states,  # .squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_mid_block = TtUNetMidBlock2DCrossAttn(**kwargs, state_dict=state_dict, base_address="mid_block")

    tt_sample = torch_to_tt_tensor_rm(sample, device, put_on_device=False)
    tt_emb = torch_to_tt_tensor_rm(emb, device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)

    tt_output = tt_mid_block(
        tt_sample,
        tt_emb,
        encoder_hidden_states=tt_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_output = tt_to_torch_tensor(tt_output)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


# lower PCC: 0.9890321746745357")
@pytest.mark.skip(reason="Test not run and is failing")
def test_run_unet_mid_block_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    test = "test1"

    if test == "test1":
        in_channels = 1280
        temb_channels = 1280
        dropout = 0.0
        num_layers = 1
        resnet_eps = 1e-05
        resnet_time_scale_shift = "default"
        resnet_act_fn = "silu"
        resnet_groups = 32
        resnet_pre_norm = True
        attn_num_head_channels = 8
        output_scale_factor = 1
        cross_attention_dim = 768
        dual_cross_attention = False
        use_linear_projection = False
        upcast_attention = False
        base_address = "mid_block"
        mid_block = pipe.unet.mid_block
        sample_shape = (2, 1280, 8, 8)
        emb_shape = (1, 1, 2, 1280)
        encoder_hidden_states_shape = (1, 2, 77, 768)
        attention_mask = None
        cross_attention_kwargs = None

    sample = torch.randn(sample_shape)
    emb = torch.randn(emb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = mid_block(
        sample,
        emb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_mid_block = TtUNetMidBlock2DCrossAttn(
        in_channels=in_channels,
        temb_channels=temb_channels,
        resnet_eps=resnet_eps,
        attn_num_head_channels=attn_num_head_channels,
        cross_attention_dim=cross_attention_dim,
        state_dict=state_dict,
        base_address="mid_block",
    )

    tt_sample = torch_to_tt_tensor_rm(sample, device, put_on_device=False)
    tt_emb = torch_to_tt_tensor_rm(emb, device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)

    tt_output = tt_mid_block(
        tt_sample,
        tt_emb,
        encoder_hidden_states=tt_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_output = tt_to_torch_tensor(tt_output)

    passing = comp_pcc(torch_output, tt_output, pcc=0.98)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
