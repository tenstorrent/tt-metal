# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_allclose,
    comp_pcc,
)


from models.experimental.ssd.tt.ssd_backbone import (
    TtSSDLiteFeatureExtractorMobileNet,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_backbone_inference(device, pcc, imagenet_sample_input, reset_seeds):
    torch_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    torch_model.eval()

    pt_model = torch_model.backbone

    config = {}
    tt_model = TtSSDLiteFeatureExtractorMobileNet(
        config,
        state_dict=torch_model.state_dict(),
        base_address=f"backbone",
        device=device,
    )
    tt_model.eval()

    # Run torch model
    torch_output = pt_model(imagenet_sample_input)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=True)
    tt_output = tt_model(tt_input)

    for key, tt_tensor in tt_output.items():
        pt_output = torch_output[key]

        # Compare outputs
        tt_output_torch = tt_to_torch_tensor(tt_tensor)

        does_pass, pcc_message = comp_pcc(pt_output, tt_output_torch, pcc)

        logger.info(comp_allclose(pt_output, tt_output_torch))
        logger.info(pcc_message)

    if does_pass:
        logger.info("SSDbackbone Passed!")

    assert does_pass, f"SSDbackbone does not meet PCC requirement {pcc}."
