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
)
from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from models.experimental.stable_diffusion.tt.transformer_2d import (
    TtBasicTransformerBlock,
    TtTransformer2DModel,
)

import pytest

"""
torch.Size([2, 4096, 320]) torch.Size([2, 77, 768]) None
#############Basic Transformer#############
only_cross_attention: False
dim: 320
num_attention_heads: 8
attention_head_dim: 40
dropout: 0.0
cross_attention_dim: 768
num_embeds_ada_norm: None
attention_bias: False
only_cross_attention: False
upcast_attention: False
norm_elementwise_affine: True
final_dropout: False
#############End of Basic Transformer#############
"""


def test_run_basic_transformer_inference(device):
    # synthesize the input
    only_cross_attention = False
    dim = 320
    num_attention_heads = 8
    attention_head_dim = 40
    dropout = 0.0
    cross_attention_dim = 768
    num_embeds_ada_norm = None
    attention_bias = False
    only_cross_attention = False
    upcast_attention = False
    norm_elementwise_affine = True
    final_dropout: False
    input_shape = [1, 2, 4096, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    input = torch.randn(input_shape) * 0.01
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    basic_transformer = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0]
    torch_output = basic_transformer(input.squeeze(0), encoder_hidden_states.squeeze(0))

    # setup tt model
    tt_input = torch_to_tt_tensor(input, device)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)

    tt_basic_transformer = TtBasicTransformerBlock(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        cross_attention_dim=cross_attention_dim,
        only_cross_attention=only_cross_attention,
        upcast_attention=False,
        device=device,
        state_dict=state_dict,
        base_address="down_blocks.0.attentions.0.transformer_blocks.0",
    )

    tt_basic_transformer.eval()
    tt_out = tt_basic_transformer(tt_input, tt_encoder_hidden_states)

    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


def test_run_transformer_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    test = "test2"
    # synthesize the input
    if test == "test1":
        # assert False, "something is off here!"
        num_attention_heads = 8
        attention_head_dim = 160
        in_channels = 1280
        out_channels = 1280
        num_layers = 1
        dropout = 0.0
        norm_num_groups = 32
        cross_attention_dim = 768
        attention_bias = False
        sample_size = None
        num_vector_embeds = None
        patch_size = None
        activation_fn = "geglu"
        num_embeds_ada_norm = None
        use_linear_projection = False
        only_cross_attention = False
        upcast_attention = False
        norm_type = "layer_norm"
        norm_elementwise_affine = True
        input_shape = [2, 1280, 16, 16]
        D = 77  # manually padded to 96
        # D = 96
        encoder_hidden_states_shape = [1, 2, D, 768]
        input = torch.randn(input_shape) * 0.01
        encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
        base_address = "up_blocks.1.attentions.0"
        transformer = pipe.unet.up_blocks[1].attentions[0]

    if test == "test2":
        num_attention_heads = 8
        attention_head_dim = 160
        in_channels = 1280
        out_channels = 1280
        num_layers = 1
        dropout = 0.0
        norm_num_groups = 32
        cross_attention_dim = 768
        attention_bias = False
        sample_size = None
        num_vector_embeds = None
        patch_size = None
        activation_fn = "geglu"
        num_embeds_ada_norm = None
        use_linear_projection = False
        only_cross_attention = False
        upcast_attention = False
        norm_type = "layer_norm"
        norm_elementwise_affine = True
        input_shape = (2, 1280, 8, 8)
        encoder_hidden_states_shape = [1, 2, 77, 768]
        input = torch.randn(input_shape) * 0.01
        encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
        base_address = "mid_block.attentions.0"
        transformer = pipe.unet.mid_block.attentions[0]

    torch_output = transformer(input, encoder_hidden_states.squeeze(0)).sample

    # setup tt model
    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)

    tt_transformer = TtTransformer2DModel(
        in_channels=in_channels,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        cross_attention_dim=cross_attention_dim,
        device=device,
        state_dict=state_dict,
        base_address=base_address,
    )
    ttnn.synchronize_device(device)
    tt_out = tt_transformer(tt_input, tt_encoder_hidden_states)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
