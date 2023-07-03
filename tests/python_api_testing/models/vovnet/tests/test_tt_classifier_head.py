from pathlib import Path
import sys
import torch
import pytest
import timm
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../..")

from models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from models.vovnet.tt.classifier_head import TtClassifierHead


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_classifier_head_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    base_address = f"head"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.head

    tt_model = TtClassifierHead(
        in_features=1024,
        num_classes=1000,
        pool_type="avg",
        use_conv=False,
        input_fmt="NCHW",
        device=device,
        host=host,
        base_address=base_address,
        state_dict=model.state_dict(),
    )

    # run torch model
    input = torch.randn(1, 1024, 7, 7)
    model_output = torch_model(input)

    # run tt model
    tt_input = torch_to_tt_tensor_rm(input, host)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output, host)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

    # compare output
    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if passing:
        logger.info("ClassifierHead Passed!")
    else:
        logger.warning("ClassifierHead Failed!")

    assert passing
