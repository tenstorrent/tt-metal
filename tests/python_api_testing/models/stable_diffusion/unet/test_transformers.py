from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional

from libs import tt_lib as ttl
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc
from transformer_2d import TtBasicTransformerBlock, TtTransformer2DModel


def test_run_basic_transformer_inference():
    # synthesize the input
    dim = 1280
    dropout = 0
    heads = 8
    bias=False
    cross_attention_dim = None,
    upcast_attention = False
    input_shape  = [1, 2, 256, 1280]
    # D = 77 # manually padded to 96
    D = 96
    encoder_hidden_states_shape  = [1, 2, D, 768]
    input = torch.randn(input_shape) * 0.01
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
    # setup pytorch model

    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    basic_transformer = pipe.unet.mid_block.attentions[0].transformer_blocks[0]
    torch_output = basic_transformer(input.squeeze(0), encoder_hidden_states.squeeze(0))

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    tt_input = torch_to_tt_tensor(input, device)

    # setup tt model
    tt_encoder_hidden_states = torch_to_tt_tensor(encoder_hidden_states, device)

    tt_basic_transformer = TtBasicTransformerBlock(
        dim = 1280,
        num_attention_heads = 8,
        attention_head_dim = 8,
        cross_attention_dim = 768,
        device=device,
        host=host,
        state_dict=state_dict,
        base_address="mid_block.attentions.0.transformer_blocks.0",)

    tt_out = tt_basic_transformer(tt_input, tt_encoder_hidden_states)
    tt_output = tt_to_torch_tensor(tt_out, host)


    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


def test_run_transformer_inference():
    # synthesize the input
    in_channels = 320
    num_attention_heads = 8
    attention_head_dim = 40
    cross_attention_dim = 768


    input_shape  = [2, 320, 64, 64]
    # D = 77 # manually padded to 96
    D = 96
    encoder_hidden_states_shape  = [1, 2, D, 768]
    input = torch.randn(input_shape) * 0.01
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    transformer = pipe.unet.down_blocks[0].attentions[0]
    torch_output = transformer(input, encoder_hidden_states.squeeze(0))

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    base_address = "down_blocks.0.attentions.0"
    tt_input = torch_to_tt_tensor(input, device)
    tt_encoder_hidden_states = torch_to_tt_tensor(encoder_hidden_states, device)

    tt_basic_transformer = TtTransformer2DModel(
        in_channels = in_channels,
        num_attention_heads = num_attention_heads,
        attention_head_dim = attention_head_dim,
        cross_attention_dim = cross_attention_dim,
        device=device,
        host=host,
        state_dict=state_dict,
        base_address=base_address,)

    tt_out = tt_basic_transformer(tt_input, tt_encoder_hidden_states)
    tt_output = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


test_run_basic_transformer_inference()
# test_run_transformer_inference()
