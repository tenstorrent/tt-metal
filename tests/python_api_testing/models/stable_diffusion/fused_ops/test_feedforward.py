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
from diffusers import StableDiffusionPipeline
from loguru import logger

from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc
from libs import tt_lib as ttl
from feedforward import TtFeedForward


def test_feedforward_inference():
    # synthesize the input
    dim = 1280
    dropout = 0
    act = "geglu"
    final_dropout = False
    input_shape  = [1, 2, 64, 1280]
    input = torch.randn(input_shape) * 0.01

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    ff = pipe.unet.mid_block.attentions[0].transformer_blocks[0].ff
    torch_output = ff(input)

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    tt_ff = TtFeedForward(dim=dim, dropout=dropout, activation_fn=act, final_dropout=False, state_dict=state_dict, device=device, host=host,)
    tt_input = torch_to_tt_tensor(input, device)
    tt_output = tt_ff(tt_input)
    tt_output = tt_to_torch_tensor(tt_output, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")

test_feedforward_inference()
