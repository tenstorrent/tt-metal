# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
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

from models.experimental.ssd.tt.ssd_classification_head import (
    TtSSDclassificationhead,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_backbone_inference(device, pcc, reset_seeds):
    torch_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    torch_model.eval()

    pt_model = torch_model.head.classification_head
    Inchannel = [672, 480, 512, 256, 256, 128]

    num_classes = 91

    config = {}
    tt_model = TtSSDclassificationhead(
        config,
        in_channels=Inchannel,
        num_classes=num_classes,
        state_dict=torch_model.state_dict(),
        base_address=f"head.classification_head.module_list",
        device=device,
    )
    tt_model.eval()

    input_tensor1 = torch.rand(1, 672, 20, 20)
    input_tensor2 = torch.rand(1, 480, 10, 10)
    input_tensor3 = torch.rand(1, 512, 5, 5)
    input_tensor4 = torch.rand(1, 256, 3, 3)
    input_tensor5 = torch.rand(1, 256, 2, 2)
    input_tensor6 = torch.rand(1, 128, 1, 1)

    tensor_list = []
    tensor_list.append(input_tensor1)
    tensor_list.append(input_tensor2)
    tensor_list.append(input_tensor3)
    tensor_list.append(input_tensor4)
    tensor_list.append(input_tensor5)
    tensor_list.append(input_tensor6)

    # Run torch model
    torch_output = pt_model(tensor_list)

    tt_tensor = []
    tt_tensor.append(input_tensor1)
    tt_tensor.append(input_tensor2)
    tt_tensor.append(input_tensor3)
    tt_tensor.append(input_tensor4)
    tt_tensor.append(input_tensor5)
    tt_tensor.append(input_tensor6)
    for i in range(len(tt_tensor)):
        if i == 2 or i == 3 or i == 5:
            tt_tensor[i] = torch_to_tt_tensor_rm(tt_tensor[i], device, put_on_device=False)
        else:
            tt_tensor[i] = torch_to_tt_tensor_rm(tt_tensor[i], device)

    tt_output = tt_model(tt_tensor)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SSDclassificationhead Passed!")

    assert does_pass, f"SSDclassificationhead output does not meet PCC requirement {pcc}."
