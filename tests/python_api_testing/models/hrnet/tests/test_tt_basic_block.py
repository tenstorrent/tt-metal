from pathlib import Path
import sys
import torch
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../../")

import timm
from tt_models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from hrnet.tt.basicblock import (
    TtBasicBlock,
)


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hrnet_w18_small", 0.99),),
)
def test_hrnet_basic_block_inference(model_name, pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    BASIC_BLOCK_LAYER_INDEX = 0
    base_address = f"stage2.{BASIC_BLOCK_LAYER_INDEX}.branches.{BASIC_BLOCK_LAYER_INDEX}.{BASIC_BLOCK_LAYER_INDEX}"
    model = timm.create_model(model_name, pretrained=True)

    # Torch BasicBlock
    torch_model = model.stage2[BASIC_BLOCK_LAYER_INDEX].branches[
        BASIC_BLOCK_LAYER_INDEX
    ][BASIC_BLOCK_LAYER_INDEX]

    # Tt BasicBlock
    tt_model = TtBasicBlock(
        in_ch=16,
        out_ch=16,
        state_dict=model.state_dict(),
        base_address=base_address,
        host=host,
        device=device,
    )

    inputs = torch.rand(1, 16, 56, 56)
    tt_inputs = torch_to_tt_tensor_rm(inputs, device)

    torch_output = torch_model(inputs)
    tt_output = tt_model(tt_inputs)

    tt_output_torch = tt_to_torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if does_pass:
        logger.info("BasicBlock Passed!")
    else:
        logger.warning("BasicBlock Failed!")

    assert does_pass
