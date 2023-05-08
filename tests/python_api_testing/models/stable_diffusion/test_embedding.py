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
from python_api_testing.models.stable_diffusion.time_embedding import TtTimestepEmbedding
from loguru import logger
# 320 1280 inputs to time embedding
# t_emb (2, 320)

def test_run_embedding_inference():
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    # synthesize the input
    timestep_input_dim = 320
    time_embed_dim = 1280
    t_emb_shape = (1, 1, 2, 320)
    t_emb = torch.randn(t_emb_shape)
    base_address="time_embedding"
    time_embedding = unet.time_embedding

    #execute torch
    torch_output = time_embedding(t_emb.squeeze(0).squeeze(0))

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup tt models
    tt_input = torch_to_tt_tensor_rm(t_emb, device, put_on_device=False)
    tt_model = TtTimestepEmbedding(timestep_input_dim,
                                    time_embed_dim,
                                    state_dict=state_dict,
                                    base_address=base_address,
                                    device=device
                                    )

    tt_model.eval()

    tt_output = tt_model(tt_input)
    tt_output = tt_to_torch_tensor(tt_output, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")


test_run_embedding_inference()
