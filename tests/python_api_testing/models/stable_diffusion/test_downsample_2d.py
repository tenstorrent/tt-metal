from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline
import numpy as np
from loguru import logger

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from utility_functions import comp_allclose_and_pcc, comp_pcc

from downsample_2d import TtDownsample2D



def test_run_downsample2d_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_downblock = pipe.unet.down_blocks[0]
    resnet_downsampler = unet_downblock.downsamplers[0]

    # synthesize the input
    input_shape =  [1, 320, 32, 32]
    input = torch.randn(input_shape)
    in_channels = 320
    out_channels = 320

    # excute pytorch
    torch_output = resnet_downsampler(input)

    # setup tt models
    tt_input = torch_to_tt_tensor(input, device)

    tt_down = TtDownsample2D(channels=in_channels, out_channels=out_channels, use_conv=True, state_dict=state_dict, base_address="down_blocks.0.downsamplers.0")
    tt_out = tt_down(tt_input)
    tt_output = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
