# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger


import ttnn
from models.utility_functions import (
    torch_to_tt_tensor,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from models.experimental.stable_diffusion.tt.cross_attention import TtCrossAttention


def test_cross_attn_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    # synthesize the input
    query_dim = 320
    dim = query_dim
    cross_attention_dim = 768
    heads = 8
    dim_head = 40
    dropout = 0.0
    bias = False
    upcast_attention = False
    upcast_softmax = False
    added_kv_proj_dim = None
    norm_num_groups = None

    base_address = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"
    cross_attn = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2

    input_shape = torch.Size([1, 2, 4096, 320])
    input = torch.randn(input_shape) * 0.01
    encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    encoder_hidden_states = encoder_hidden_states.squeeze(0) if encoder_hidden_states is not None else None
    torch_output = cross_attn(input.squeeze(0), encoder_hidden_states)

    # setup tt model
    tt_cross_attn = TtCrossAttention(
        query_dim=query_dim,
        heads=heads,
        bias=bias,
        dim_head=dim_head,
        cross_attention_dim=cross_attention_dim,
        upcast_attention=upcast_attention,
        state_dict=state_dict,
        device=device,
        # host=host,
        base_address=base_address,
    )
    ttnn.synchronize_device(device)
    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_encoder_hidden_states = (
        torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)
        if encoder_hidden_states is not None
        else None
    )
    tt_out = tt_cross_attn(tt_input, tt_encoder_hidden_states)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
