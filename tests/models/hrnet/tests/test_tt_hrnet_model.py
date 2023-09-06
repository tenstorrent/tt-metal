# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../../")

import timm
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from hrnet.tt.hrnet_model import hrnet_w18_small


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hrnet_w18_small", 0.99),),
)
def test_hrnet_model_inference(model_name, pcc, imagenet_sample_input, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    torch_model = timm.create_model(model_name, pretrained=True)

    tt_model = hrnet_w18_small(device, host, multi_scale_output=True)

    torch_output = torch_model(imagenet_sample_input)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=False)
    tt_output = tt_model(tt_input)

    tt_output_torch = tt_to_torch_tensor(tt_output).view(1, -1)

    passing, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)
    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if passing:
        logger.info("HRNetForImageClassification Passed!")
    else:
        logger.warning("HRNetForImageClassification Failed!")

    assert passing
