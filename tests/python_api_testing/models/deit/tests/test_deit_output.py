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
from deit_output import TtDeiTOutput


def test_deit_output_inference():
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'encoder.layer.0.output'
    torch_output = model.encoder.layer[0].output

    hidden_state_shape =  torch.Size([1, 198, 3072])
    hidden_state = torch.randn(hidden_state_shape)

    input_tensor_shape =  torch.Size([1, 198, 768])
    input_tensor = torch.randn(input_tensor_shape)

    torch_out = torch_output(hidden_state, input_tensor)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # setup tt model

    tt_output = TtDeiTOutput(DeiTConfig(), host, device, state_dict, base_address)

    tt_hidden_state = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_input_tensor = torch_to_tt_tensor_rm(input_tensor, device, put_on_device=False)

    tt_out = tt_output(tt_hidden_state, tt_input_tensor)
    tt_out = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_out, tt_out)
    logger.info(comp_allclose_and_pcc(tt_out, torch_out))
    tt_lib.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
