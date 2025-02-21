# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
import ttnn
from models.experimental.functional_vovnet.tt.classifier_head import TtClassifierHead
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_classifier_head_inference(device, pcc, reset_seeds):
    base_address = f"head"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=Flase)

    torch_model = model.head
    print(torch_model)
    tt_model = TtClassifierHead(
        base_address=f"head",
        device=device,
        torch_model=model.state_dict(),
    )

    # run torch model
    input = torch.randn(1, 1024, 7, 7)
    model_output = torch_model(input)

    # run tt model
    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

    assert_with_pcc(model_output, tt_output_torch, 0.99)
