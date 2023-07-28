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


def test_deit_output_inference(pcc=0.99):

    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'encoder.layer.0.output'
    torch_model = model.encoder.layer[0].output

    hidden_state_shape =  torch.Size([1, 198, 3072])
    hidden_state = torch.randn(hidden_state_shape)

    input_tensor_shape =  torch.Size([1, 198, 768])
    input_tensor = torch.randn(input_tensor_shape)

    torch_output = torch_model(hidden_state, input_tensor)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    # setup tt model
    tt_output = TtDeiTOutput(DeiTConfig(), device, state_dict, base_address)

    tt_hidden_state = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_input_tensor = torch_to_tt_tensor_rm(input_tensor, device, put_on_device=False)

    tt_output = tt_output(tt_hidden_state, tt_input_tensor)
    tt_output = tt_to_torch_tensor(tt_output).squeeze(0)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    tt_lib.device.CloseDevice(device)
    assert(pcc_passing), f"Failed! Low pcc: {pcc}."
