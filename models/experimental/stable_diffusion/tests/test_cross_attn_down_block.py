# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from loguru import logger
from functools import wraps
import pytest


import ttnn
from models.utility_functions import (
    torch_to_tt_tensor,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose_and_pcc,
    skip_for_wormhole_b0,
)
from models.experimental.stable_diffusion.tt.unet_2d_blocks import TtCrossAttnDownBlock2D
from models.experimental.stable_diffusion.tt.experimental_ops import UseDeviceConv


@skip_for_wormhole_b0()
@pytest.mark.skip(reason="Test is failing, see issue #7536")
@pytest.mark.parametrize("index", [1])  # FIXME: failing 0, 2 with L1 error.
def test_run_cross_attn_down_block_real_input_inference(device, index, model_location_generator):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    dir_path = model_location_generator("tensor_files", model_subdir="StableDiffusion")
    attr_path = f"{dir_path}/CrossAttnDownBlock2D_inp__attr__block_{index}.pt"
    attention_mask_path = f"{dir_path}/CrossAttnDownBlock2D_inp__attention_mask__block_{index}.pt"
    cross_attn_kwargs_path = f"{dir_path}/CrossAttnDownBlock2D_inp__cross_attention_kwargs__block_{index}.pt"
    emb_path = f"{dir_path}/CrossAttnDownBlock2D_inp__emb__block_{index}.pt"
    encoder_hidden_states_path = f"{dir_path}/CrossAttnDownBlock2D_inp__encoder_hidden_states__block_{index}.pt"
    sample_path = f"{dir_path}/CrossAttnDownBlock2D_inp__sample__block_{index}.pt"

    map_location = torch.device("cpu")
    sample = torch.load(sample_path, map_location=map_location)
    emb = torch.load(emb_path, map_location=map_location)
    encoder_hidden_states = torch.load(encoder_hidden_states_path, map_location=map_location)
    attention_mask = torch.load(attention_mask_path, map_location=map_location)
    cross_attention_kwargs = torch.load(cross_attn_kwargs_path, map_location=map_location)

    kwargs = torch.load(attr_path)
    base_address = f"down_block.{index}"
    down_block = pipe.unet.down_blocks[index]

    torch_output, torch_list_out = down_block(
        sample,
        emb,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_sample = torch_to_tt_tensor_rm(sample, device, put_on_device=False)
    tt_emb = torch_to_tt_tensor_rm(emb.unsqueeze(0).unsqueeze(0), device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states.unsqueeze(0), device, put_on_device=False)

    tt_cross_attn_down_block = TtCrossAttnDownBlock2D(
        **kwargs, state_dict=state_dict, base_address=f"down_blocks.{index}"
    )

    tt_output, list_out = tt_cross_attn_down_block(
        tt_sample,
        tt_emb,
        encoder_hidden_states=tt_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_output = tt_to_torch_tensor(tt_output)

    for _tt, _torch in zip(list_out, torch_list_out):
        tt_o = tt_to_torch_tensor(_tt)
        logger.info(comp_allclose_and_pcc(_torch, tt_o))

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


@pytest.mark.skip(reason="Test not run and failing")
def test_run_cross_attn_down_block_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    test = "test1"

    if test == "test1":
        in_channels = 320
        out_channels = 320
        temb_channels = 1280
        dropout = 0.0
        num_layers = 2
        resnet_eps = 1e-05
        resnet_time_scale_shift = "default"
        resnet_act_fn = "silu"
        resnet_groups = 32
        resnet_pre_norm = True
        attn_num_head_channels = 8
        cross_attention_dim = 768
        output_scale_factor = 1.0
        downsample_padding = 1
        add_downsample = True
        dual_cross_attention = False
        use_linear_projection = False
        only_cross_attention = False
        upcast_attention = False
        sample_shape = [2, 320, 64, 64]
        temb_shape = [1, 1, 2, 1280]  # original: 2, 1280
        encoder_hidden_states_shape = [1, 2, 77, 768]  # original: 2, 77, 768
        attention_mask = None
        cross_attention_kwargs = None
        base_address = "down_blocks.0"
        down_block = pipe.unet.down_blocks[0]

    sample = torch.randn(sample_shape)
    emb = torch.randn(temb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output, torch_list_out = down_block(
        sample,
        emb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    tt_cross_attn_down_block = TtCrossAttnDownBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        dropout=dropout,
        num_layers=num_layers,
        resnet_eps=resnet_eps,
        resnet_time_scale_shift=resnet_time_scale_shift,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        resnet_pre_norm=resnet_pre_norm,
        attn_num_head_channels=attn_num_head_channels,
        cross_attention_dim=cross_attention_dim,
        output_scale_factor=output_scale_factor,
        downsample_padding=downsample_padding,
        add_downsample=add_downsample,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        only_cross_attention=only_cross_attention,
        upcast_attention=upcast_attention,
        state_dict=state_dict,
        base_address="down_blocks.0",
    )

    tt_sample = torch_to_tt_tensor_rm(sample, device, put_on_device=False)
    tt_emb = torch_to_tt_tensor_rm(emb, device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)

    tt_output, list_out = tt_cross_attn_down_block(
        tt_sample,
        tt_emb,
        encoder_hidden_states=tt_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    ttnn.synchronize_device(device)
    tt_output = tt_to_torch_tensor(tt_output)

    passing = comp_pcc(torch_output, tt_output, pcc=0.95)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
