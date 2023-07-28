from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional, Set, Tuple, Union

import torch
from torch import nn
from loguru import logger


import tt_lib
from utility_functions_new import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor
from utility_functions_new import comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig

from transformers import DeiTModel
from deit_intermediate import TtDeiTIntermediate


def test_deit_intermediate_inference(pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'encoder.layer.0.intermediate'
    torch_intermediate = model.encoder.layer[0].intermediate

    input_shape =  torch.Size([1, 198, 768])
    hidden_state = torch.randn(input_shape)

    torch_output = torch_intermediate(hidden_state)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # setup tt model
    tt_intermediate = TtDeiTIntermediate(DeiTConfig(), device, state_dict, base_address)

    tt_input = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_out = tt_intermediate(tt_input)
    tt_output = tt_to_torch_tensor(tt_out).squeeze(0)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    tt_lib.device.CloseDevice(device)
    assert(pcc_passing), f"Failed! Low pcc: {pcc}."
