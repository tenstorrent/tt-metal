from pathlib import Path
import sys
import torch
import pytest
import timm
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from tt_models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from python_api_testing.models.vovnet.tt.sequential_append_list import (
    TtSequentialAppendList,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_sequential_append_list_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    STAGE_INDEX = 0
    BLOCK_INDEX = 0

    base_address = f"stages.{STAGE_INDEX}.blocks.{BLOCK_INDEX}"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stages[STAGE_INDEX].blocks[BLOCK_INDEX].conv_mid

    tt_model = TtSequentialAppendList(
        in_channels=128,
        groups=128,
        layer_per_block=3,
        state_dict=model.state_dict(),
        base_address=f"{base_address}",
    )

    input = torch.randn(1, 128, 56, 56)
    model_output = torch_model(input, [input])

    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input, [tt_input])
    tt_output_torch = tt_to_torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0)

    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if passing:
        logger.info("SequntialAppendList Passed!")
    else:
        logger.warning("SequntialAppendList Failed!")

    assert passing
