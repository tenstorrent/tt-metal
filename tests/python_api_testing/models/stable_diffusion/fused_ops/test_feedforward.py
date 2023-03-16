from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from loguru import logger
from diffusers import StableDiffusionPipeline
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from libs import tt_lib as ttl

import torch


def run_test_feed_forward_inference(device, host):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    ff = pipe.unet.mid_block.attentions[0].transformer_blocks[0].ff
    print(ff.net[2])

    dim = 1280
    dropout = 0
    act = "geglu"
    final_dropout = False

    input_shape  = [1, 2, 64, 1280]

    input = torch.randn(input_shape) * 0.01
    torch_out = ff(input)

    tt_input = torch_to_tt_tensor(input, device)
    tt_ff = TtFeedForward(dim=dim, dropout=dropout, activation_fn=act, final_dropout=False, state_dict=state_dict, device=device, host=host,)
    tt_out = tt_ff(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)

    does_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.99)

    print(comp_allclose(torch_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("test_FeedForward_inference Passed!")
    else:
        logger.warning("test_FeedForward_inference Failed!")
    assert does_pass


def test_feedforward_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_feed_forward_inference(device, host)
    ttl.device.CloseDevice(device)
