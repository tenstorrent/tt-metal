from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig
from transformers import DeiTModel
from deit_patch_embeddings import TtDeiTPatchEmbeddings

def test_deit_patch_embeddings_inference():
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'embeddings.patch_embeddings.projection'
    torch_patch_embeddings = model.embeddings.patch_embeddings

    input_shape =  torch.Size([1, 3, 224, 224])
    pixel_values = torch.randn(input_shape)

    torch_output = torch_patch_embeddings(pixel_values)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # setup tt model

    tt_patch_embeddings = TtDeiTPatchEmbeddings(DeiTConfig(), host, device, state_dict, base_address)

    tt_input = torch_to_tt_tensor_rm(pixel_values, device, put_on_device=False)
    tt_out = tt_patch_embeddings(tt_input)
    tt_output = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    tt_lib.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
