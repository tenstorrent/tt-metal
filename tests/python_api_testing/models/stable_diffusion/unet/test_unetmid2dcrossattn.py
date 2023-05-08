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
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from utility_functions import comp_pcc, comp_allclose_and_pcc, torch_to_tt_tensor_rm
from unet_2d_blocks import TtUNetMidBlock2DCrossAttn

from loguru import logger


def test_run_unetmid2dcrossattn_inference():
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
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
        resnet_time_scale_shift = 'default'
        resnet_act_fn = 'silu'
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
        cross_attention_kwargs=None

    sample = torch.randn(sample_shape)
    emb = torch.randn(emb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = mid_block(
                        sample,
                        emb.squeeze(0).squeeze(0),
                        encoder_hidden_states=encoder_hidden_states.squeeze(0),
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs
                        )

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    tt_mid_block = TtUNetMidBlock2DCrossAttn(
                    in_channels=in_channels,
                    temb_channels=temb_channels,
                    resnet_eps = resnet_eps,
                    attn_num_head_channels=attn_num_head_channels,
                    cross_attention_dim=cross_attention_dim,
                    state_dict=state_dict,
                    base_address="mid_block"
    )

    tt_sample = torch_to_tt_tensor_rm(sample, device, put_on_device=False)
    tt_emb = torch_to_tt_tensor_rm(emb, device, put_on_device=False)
    tt_encoder_hidden_states = torch_to_tt_tensor_rm(encoder_hidden_states, device, put_on_device=False)

    tt_output = tt_mid_block(
        tt_sample,
        tt_emb,
        encoder_hidden_states=tt_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs
        )

    tt_output = tt_to_torch_tensor(tt_output, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")

test_run_unetmid2dcrossattn_inference()
# sample shape torch.Size([2, 1280, 8, 8])
# emb shape torch.Size([2, 1280])
# encoder hidden state shape torch.Size([2, 77, 768])
# attention mask None
# cross attentino kwargs None
