from pathlib import Path
import sys
import torch
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from python_api_testing.models.swin.tt.swin_model import TtSwinModel
from python_api_testing.models.swin.swin_utils import get_shape
from transformers import SwinModel
from transformers import AutoFeatureExtractor


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_model_inference(imagenet_sample_input, pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    image = imagenet_sample_input

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224"
    )
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    inputs = feature_extractor(images=image, return_tensors="pt")

    base_address = f""

    with torch.no_grad():
        torch_model = model

        tt_model = TtSwinModel(
            config=model.config,
            state_dict=model.state_dict(),
            base_address=base_address,
            device=device,
            host=host,
        )

        # Run torch model
        torch_output = torch_model(**inputs)

        # Run tt model
        tt_pixel_values = inputs.pixel_values
        tt_pixel_values = torch_to_tt_tensor_rm(tt_pixel_values, device)
        tt_output = tt_model(tt_pixel_values)

        tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state, host)
        tt_output_torch = tt_output_torch.squeeze(0)

        does_pass, pcc_message = comp_pcc(
            torch_output.last_hidden_state, tt_output_torch, pcc
        )

        logger.info(comp_allclose(torch_output.last_hidden_state, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)
        if does_pass:
            logger.info("SwinModel Passed!")
        else:
            logger.warning("SwinModel Failed!")

        assert does_pass
