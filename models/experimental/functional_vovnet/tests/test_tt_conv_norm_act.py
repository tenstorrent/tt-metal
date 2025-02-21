# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
import ttnn
from models.experimental.functional_vovnet.tt.conv_norm_act import TtConvNormAct
from models.experimental.functional_vovnet.tt.model_preprocessing import create_vovnet_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnet_conv_norm_act_inference(device, pcc, reset_seeds):
    base_address = f"stem.0"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    for k, v in model.state_dict().items():
        print(k)
    # parameters = create_vovnet_model_parameters(model, device=device)
    torch_model = model.stem[0]
    tt_model = TtConvNormAct(
        # kernel_size=3,
        stride=2,
        base_address=base_address,
        device=device,
        torch_model=model.state_dict(),
        # input_params = [1, 224, 224, 3],
        split_conv=False,
    )

    # run torch model
    input = torch.rand(1, 3, 224, 224)
    model_output = torch_model(input)

    # run tt model
    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output[0])
    # tt_output_torch = torch.permute(tt_output_torch, (0, 3, 1, 2))
    # tt_output_torch = torch.reshape(tt_output_torch, model_output.shape)
    assert_with_pcc(model_output, tt_output_torch, 0.99)
    # compare output
