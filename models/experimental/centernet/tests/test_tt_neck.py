# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_centernet_neck import TtCTResNetNeck
from models.experimental.centernet.reference.ct_resnet_neck import CTResNetNeck
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet_neck(device, reset_seeds):
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if "neck" in k:
            new_state_dict[k] = v
    with torch.no_grad():
        reference = CTResNetNeck(parameters=new_state_dict)
        reference.eval()

    parameters = custom_preprocessor(device, state_dict)

    input1 = torch.rand((1, 64, 112, 168))
    input2 = torch.rand((1, 128, 56, 84))
    input3 = torch.rand((1, 256, 28, 42))
    input4 = torch.rand((1, 512, 14, 21))
    input = (input1, input2, input3, input4)
    torch_output = reference.forward(input)

    Neck = TtCTResNetNeck(parameters=parameters, device=device)

    tt_input = (
        ttnn.from_torch(input1, dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(input2, dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(input3, dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(input4, dtype=ttnn.bfloat16, device=device),
    )

    tt_output = Neck.forward(tt_input)

    assert_with_pcc(ttnn.to_torch(tt_output[0]), torch_output[0], 0.99)
