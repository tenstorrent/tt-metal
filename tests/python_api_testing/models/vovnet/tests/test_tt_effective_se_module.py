from pathlib import Path
import sys
import torch
import pytest
import timm
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib

from tt.effective_se_module import (
    TtEffectiveSEModule,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_effective_se_module_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    base_address = f"stages.0.blocks.0.attn"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stages[0].blocks[0].attn

    tt_model = TtEffectiveSEModule(
        in_channels=256,
        out_channels=256,
        kernel_size=1,
        stride=1,
        dilation=1,
        padding=0,
        bias=None,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
        host=host,
    )

    # run torch model
    input = torch.randn(1, 256, 56, 56)
    model_output = torch_model(input)

    # run tt model
    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output, host)
    tt_output_torch = tt_output_torch.squeeze(0)

    # compare output
    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if passing:
        logger.info("EffectiveSEModule Passed!")
    else:
        logger.warning("EffectiveSEModule Failed!")

    assert passing
