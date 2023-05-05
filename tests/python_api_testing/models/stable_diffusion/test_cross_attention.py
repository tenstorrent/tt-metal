from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger

from libs import tt_lib as ttl
from utility_functions import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor
from utility_functions import comp_pcc, comp_allclose_and_pcc

from cross_attention import TtCrossAttention


def test_cross_attn_inference():

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    test = "test1"
    # synthesize the input
    if test == "test1":
        dim = 1280
        dropout = 0
        heads = 8
        dim_head = 64
        bias=False
        cross_attention_dim = None
        upcast_attention = False
        input_shape  = [1, 2, 64, 1280]
        input = torch.randn(input_shape) * 0.01
        encoder_hidden_states = None
        # base_address = "ISLOST!"
        base_address="mid_block.attentions.0.transformer_blocks.0.attn1"
        cross_attn = pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1


    ##############################################
    if test == "test2":
        assert False, "this test doesn't work right now!"
        dim = 1280
        heads = 8
        dim_head = 160
        cross_attention_dim = 768
        dropout = 0
        bias = False
        cross_attention_dim = None
        upcast_attention = False
        dim_head = 64
        input_shape = (1, 2, 256, 1280)
        encoder_hidden_states_shape = (1, 2, 77, 768)
        input = torch.randn(input_shape)
        encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
        # base_address="mid_block.attentions.0.transformer_blocks.0.attn1"
        base_address="down_blocks.0.attentions.0.transformer_blocks.0.attn1"
        cross_attn = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1


    if test == "test3":
        dim = 320
        cross_attention_dim = 768
        heads = 8
        dim_head = 40
        dropout  = 0.0
        bias = False
        upcast_attention = False
        upcast_softmax = False
        added_kv_proj_dim = None
        norm_num_groups = None
        input_shape = (1, 2, 4096, 320)
        encoder_hidden_states_shape = (1, 2, 77, 768)
        input = torch.randn(input_shape)
        encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
        base_address="down_blocks.0.attentions.0.transformer_blocks.0.attn2"
        cross_attn = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2


    print(cross_attn)
    encoder_hidden_states = encoder_hidden_states.squeeze(0) if encoder_hidden_states is not None else None
    torch_output = cross_attn(input.squeeze(0), encoder_hidden_states)

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    tt_cross_attn = TtCrossAttention(query_dim=dim,
                                    heads = heads,
                                    bias=bias,
                                    dim_head=dim_head,
                                    cross_attention_dim=cross_attention_dim,
                                    upcast_attention=upcast_attention,
                                    state_dict=state_dict,
                                    device=device,
                                    host=host,
                                    base_address=base_address)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False) if encoder_hidden_states is not None else None
    tt_out = tt_cross_attn(tt_input, tt_encoder_hidden_states)
    tt_output = tt_to_torch_tensor(tt_out, host)


    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


test_cross_attn_inference()
