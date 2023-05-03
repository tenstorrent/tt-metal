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


from libs import tt_lib as ttl
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor

from cross_attention import TtCrossAttention

    #     query_dim: int,
    #     cross_attention_dim: Optional[int] = None,
    #     heads: int = 8,
    #     dim_head: int = 64,
    #     dropout: float = 0.0,
    #     bias=False,
    #     upcast_attention: bool = False,
    #     upcast_softmax: bool = False,
    #     added_kv_proj_dim: Optional[int] = None,
    #     norm_num_groups: Optional[int] = None,
    #     processor: Optional["AttnProcessor"] = None,
    #     device=None,
    #     host=None,
    #     state_dict=None,
    #     base_address="mid_block.attentions.0.transformer_blocks.0.attn1",


def test_cross_attn_inference():
    # synthesize the input
    dim = 1280
    dropout = 0
    heads = 8
    bias=False
    cross_attention_dim = None
    upcast_attention = False
    input_shape  = [2, 64, 1280]
    input = torch.randn(input_shape) * 0.01

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    cross_attn = pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    torch_output = cross_attn(input)

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    tt_cross_attn = TtCrossAttention(query_dim=dim,
                                    heads = heads,
                                    bias=False,
                                    cross_attention_dim=cross_attention_dim,
                                    upcast_attention=upcast_attention,
                                    state_dict=state_dict,
                                    device=device,
                                    host=host,)

    tt_input = torch_to_tt_tensor(input, device)
    tt_out = tt_cross_attn(tt_input)
    tt_output = tt_to_torch_tensor(tt_out, host)


    from utility_functions import comp_pcc, comp_allclose_and_pcc
    from loguru import logger


    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


test_cross_attn_inference()
