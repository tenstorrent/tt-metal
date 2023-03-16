from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from diffusers import StableDiffusionPipeline
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor

from libs import tt_lib as ttl
import torch


def run_test_up_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    up_block = pipe.unet.up_blocks[0]

    # print(up_block)
    in_channels = 1280
    out_channels = 1280
    num_layers = 3
    resnet_eps = 1e-5
    prev_output_channel = 1280
    out_channels = 1280
    temb_channels = 1280

    input_shape  = [2, 1280, 8, 8]
    input = torch.randn(input_shape) * 0.01
    input2 = (torch.randn(input_shape), torch.randn(input_shape))
    temb = torch.randn([2, 1280])

    torch_out = up_block(input, input2, temb)
    # TODO: inputs given to up_block are correct sized, but
    # we need to pad them before feeding them to ttUpBlock
    tt_input = torch_to_tt_tensor(input, device)

    tt_ub = TtUpBlock2D(
        in_channels = in_channels,
        prev_output_channel = prev_output_channel,
        out_channels = out_channels,
        temb_channels  = temb_channels,
        num_layers = num_layers,
        resnet_eps = resnet_eps,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    )

    tt_out = tt_ub(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)

    oes_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.99)

    print(comp_allclose(torch_out, tt_out))
    print(pcc_message)


    if does_pass:
        logger.info("test_UPBLOCK_inference Passed!")
    else:
        logger.warning("test_UPBLOCK_inference Failed!")
    assert does_pass



def run_test_down_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    down_block = pipe.unet.down_blocks[3]
    prompt = "something"

    pipe(prompt, num_inference_steps=1)
    assert False

    print(down_block)
    in_channels =  1280
    out_channels =  1280
    temb_channels =  1280
    num_layers =  2
    resnet_eps =  1e-05
    resnet_groups =  32



    input_shape  = [2, 1280, 8, 8]

    input = torch.randn(input_shape) * 0.01
    torch_out = down_block(input)

    tt_input = torch_to_tt_tensor(input, device)

    tt_db = TtDownBlock2D(
        in_channels = in_channels,
        out_channels = out_channels,
        temb_channels = temb_channels,
        num_layers = 2,
        resnet_eps= 1e-5,
        downsample_padding=1,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    )

    tt_out = tt_db(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)

    oes_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.99)

    print(comp_allclose(torch_out, tt_out))
    print(pcc_message)


    if does_pass:
        logger.info("test_DOWNBLOCK_inference Passed!")
    else:
        logger.warning("test_DOWNBLOCK_inference Failed!")
    assert does_pass



def test_up_block_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_up_block_inference(device)
    ttl.device.CloseDevice(device)


def test_down_block_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_down_block_inference(device)
    ttl.device.CloseDevice(device)

test_down_block_inference()
# test_up_block_inference()
