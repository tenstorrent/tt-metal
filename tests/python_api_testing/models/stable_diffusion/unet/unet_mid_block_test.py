from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from diffusers import StableDiffusionPipeline

from python_api_testing.models.stable_diffusion.unet.unet_2d_blocks import TtUNetMidBlock2D, TtDownEncoderBlock2D, TtUpDecoderBlock2D
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


def run_up_decoder_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    block_id = 0
    up_block = pipe.vae.decoder.up_blocks[block_id]

    num_layers = 2
    resnet_act_fn = "silu"

    if block_id == 0:
        in_channels = 512
        out_channels = 512
        add_upsample = True
        input_shape  = [1, 512, 64, 64]

    input = torch.randn(input_shape)
    torch_out = up_block(input)

    tt_input = torch_to_tt_tensor(input, device)
    tt_up_block = TtUpDecoderBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                add_upsample=add_upsample,
                num_layers=num_layers,
                resnet_act_fn=resnet_act_fn,

                state_dict=state_dict,
                device=device,
                host=host,
                base_address=f"decoder.up_blocks.{block_id}")
    print(tt_up_block)
    tt_out = tt_up_block(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))
    print("up decoder executed successfully")



def run_down_encoder_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    block_id = 1
    down_block = pipe.vae.encoder.down_blocks[block_id]

    num_layers = 2
    resnet_act_fn = "silu"
    downsample_padding = 0

    if block_id == 1:
        in_channels = 128
        out_channels = 256
        add_downsample = True
        input_shape  = [1, 128, 64, 64]

    if block_id == 2:
        in_channels = 256
        out_channels = 512
        add_downsample = True
        input_shape  = [1, 256, 64, 64]

    input = torch.randn(input_shape)
    torch_out = down_block(input)

    tt_input = torch_to_tt_tensor(input, device)
    tt_down_block = TtDownEncoderBlock2D(in_channels=in_channels,
                out_channels=out_channels,
                add_downsample=add_downsample,
                num_layers=num_layers,
                resnet_act_fn=resnet_act_fn,
                downsample_padding=downsample_padding,
                state_dict=state_dict,
                device=device,
                host=host,
                base_address=f"encoder.down_blocks.{block_id}")

    tt_out = tt_down_block(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))
    print("down block executed successfully")


def run_mid_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_encoder = pipe.vae.encoder
    mid_block = vae_encoder.mid_block


    in_channels = 512
    eps = 1e-06
    resnet_groups = 32
    input_shape  = [1, 512, 64, 64]

    input = torch.randn(input_shape)
    torch_out = mid_block(input, None)

    tt_input = torch_to_tt_tensor(input, device)
    tt_mid_block = TtUNetMidBlock2D(in_channels=in_channels, temb_channels=None, resnet_act_fn="silu", attn_num_head_channels=1, state_dict=state_dict, device=device, host=host,)
    tt_out = tt_mid_block(tt_input, None)
    tt_out = tt_to_torch(tt_out, host)

    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))
    print("mid block executed successfully")


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    # run_mid_block_inference(device)
    # run_down_encoder_block_inference(device)
    run_up_decoder_block_inference(device)

    ttl.device.CloseDevice(device)
