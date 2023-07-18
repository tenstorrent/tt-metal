import torch
import pytest
from loguru import logger

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose,
    comp_pcc,
)
from models.ssd.tt.ssd_mobilenetv3_features import (
    TtMobileNetV3Features,
)

import tt_lib


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_mobilenetv3_features_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    torch_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    torch_model.eval()

    FEATURE_INDEX = 0
    pt_model = torch_model.backbone.features[FEATURE_INDEX]

    config = {}
    tt_model = TtMobileNetV3Features(
        config,
        state_dict=torch_model.state_dict(),
        base_address=f"backbone.features",
        device=device,
    )

    # Run torch model
    input_tensor = torch.rand(1, 3, 320, 320)
    torch_output = pt_model(input_tensor)

    # Run tt model
    tt_mobilenet_input = torch_to_tt_tensor_rm(input_tensor, device)
    tt_output = tt_model(tt_mobilenet_input)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output)
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SSDMobilenetV3Features Passed!")

    assert does_pass, f"SSDMobilenetV3Features does not meet PCC requirement {pcc}."
