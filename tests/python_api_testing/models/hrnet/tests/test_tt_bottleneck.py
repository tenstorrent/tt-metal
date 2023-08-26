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
from hrnet.tt.bottleneck import (
    TtBottleneck,
)


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hrnet_w18_small", 0.99),),
)
def test_hrnet_bottleneck_inference(model_name, pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    BOTTLENECK_LAYER_INDEX = 0
    base_address = f"layer1.{BOTTLENECK_LAYER_INDEX}"
    model = timm.create_model(model_name, pretrained=True)

    # Torch Bottleneck
    torch_model = model.layer1[BOTTLENECK_LAYER_INDEX]

    # Tt Bottleneck
    tt_model = TtBottleneck(
        in_ch=64,
        out_ch=32,
        state_dict=model.state_dict(),
        base_address=base_address,
        host=host,
        device=device,
    )

    inputs = torch.rand(1, 64, 56, 56)
    tt_inputs = torch_to_tt_tensor_rm(inputs, device)

    torch_output = torch_model(inputs)
    tt_output = tt_model(tt_inputs)

    tt_output_torch = tt_to_torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if does_pass:
        logger.info("Bottleneck Passed!")
    else:
        logger.warning("Bottleneck Failed!")

    assert does_pass
