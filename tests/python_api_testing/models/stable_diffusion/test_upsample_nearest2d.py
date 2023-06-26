from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch
from diffusers import StableDiffusionPipeline
from loguru import logger

# from libs import tt_lib as ttl
import tt_lib as ttl
from utility_functions_new import torch_to_tt_tensor, tt_to_torch_tensor
from utility_functions_new import comp_pcc, comp_allclose_and_pcc, torch_to_tt_tensor_rm
from upsample_nearest2d import TtUpsampleNearest2d


def test_run_upsample_nearest_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # synthesize the input
    input_shape =  [1, 1, 32, 32]
    input = torch.randn(input_shape)

    torch_output = F.interpolate(input, scale_factor=2.0, mode="nearest")

    tt_input = torch_to_tt_tensor(input, device)
    tt_up = TtUpsampleNearest2d(scale_factor=2.0)
    tt_out = tt_up(tt_input)
    tt_output = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
