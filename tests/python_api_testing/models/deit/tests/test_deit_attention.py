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
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor
from utility_functions_new import comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig

from transformers import DeiTModel
from deit_attention import TtDeiTAttention


def test_deit_attention_inference(pcc=0.99):

    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'encoder.layer.0.attention'
    torch_attention = model.encoder.layer[0].attention
    head_mask = None
    output_attentions = False
    input_shape =  torch.Size([1, 1, 198, 768])
    hidden_state = torch.randn(input_shape)

    torch_output = torch_attention(hidden_state.squeeze(0),
                                    head_mask,
                                    output_attentions)[0]

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # setup tt model
    tt_attention = TtDeiTAttention(DeiTConfig(),
                                    device,
                                    state_dict,
                                    base_address)

    tt_input = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_out = tt_attention(tt_input, head_mask, output_attentions)[0]
    tt_output = tt_to_torch_tensor(tt_out).squeeze(0)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    tt_lib.device.CloseDevice(device)
    assert(pcc_passing), f"Failed! Low pcc: {pcc}."
