import torch
import pytest
from loguru import logger

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose,
    comp_pcc,
)
from models.ssd.tt.ssd_mobilenetv3_inverted_squeeze import (
    TtMobileNetV3InvertedSqueeze,
)
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)
import tt_lib


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_inverted_squeeze_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    TV_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    TV_model.eval()

    FEATURE_INDEX = 0
    LAYER_INDEX = 4
    # torch invertedsqueeze
    torch_model = TV_model.backbone.features[FEATURE_INDEX][LAYER_INDEX]

    # Tt ssd_invertedsqueeze
    config = {"in_channels": 24}
    tt_model = TtMobileNetV3InvertedSqueeze(
        config,
        in_channels=config["in_channels"],
        expanded_channels=72,
        out_channels=40,
        fc_channels=24,
        kernel_size=5,
        stride=2,
        padding=2,
        use_activation=True,
        state_dict=TV_model.state_dict(),
        base_address=f"backbone.features.{FEATURE_INDEX}.{LAYER_INDEX}",
        device=device,
    )

    # Run torch model
    input_tensor = torch.randn(1, 24, 56, 56)
    torch_output = torch_model(input_tensor)

    # Run tt model
    tt_inverted_squeeze_input = torch_to_tt_tensor_rm(input_tensor, device)
    tt_output = tt_model(tt_inverted_squeeze_input)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SSDInvertedSqueeze Passed!")

    assert does_pass, f"SSDInvertedSqueeze does not meet PCC requirement {pcc}."
