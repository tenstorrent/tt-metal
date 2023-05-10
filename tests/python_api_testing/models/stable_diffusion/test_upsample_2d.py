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

from upsample_2d import TtUpsample2D


def test_run_upsample2d_inference():
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]
    resnet_upsampler = unet_upblock.upsamplers[0]

    input_shape =  [1, 1280, 32, 32]
    input = torch.randn(input_shape)
    in_channels = 1280
    out_channels = 1280
    torch_output = resnet_upsampler(input)


    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    tt_input = torch_to_tt_tensor(input, device)

    tt_up = TtUpsample2D(channels=in_channels,
                        out_channels=out_channels,
                        use_conv=True,
                        use_conv_transpose=False,
                        name="conv",
                        state_dict=state_dict,
                        base_address="up_blocks.0.upsamplers.0")
    tt_out = tt_up(tt_input)


    tt_output = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
