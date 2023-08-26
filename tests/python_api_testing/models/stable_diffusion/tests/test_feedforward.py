import torch
from diffusers import StableDiffusionPipeline
from loguru import logger

from tt_models.utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from tt_models.utility_functions import comp_pcc, comp_allclose_and_pcc

import tt_lib as ttl
from tt_models.stable_diffusion.tt.feedforward import TtFeedForward


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


    # setup tt model
    tt_ff = TtFeedForward(dim=dim, dropout=dropout, activation_fn=act, final_dropout=False, state_dict=state_dict, device=device)
    ttl.device.Synchronize()
    tt_input = torch_to_tt_tensor(input, device)
    tt_output = tt_ff(tt_input)
    tt_output = tt_to_torch_tensor(tt_output)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
