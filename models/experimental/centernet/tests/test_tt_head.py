# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_centernet_head import TtCTResNetHead
from models.experimental.centernet.reference.centernet_head import CTResNetHead
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet_head(device, reset_seeds):
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]

    new_state_dict = {}

    for k, v in state_dict.items():
        if "bbox_head" in k:
            new_state_dict[k] = v

    with torch.no_grad():
        reference = CTResNetHead(parameters=new_state_dict)
        reference.eval()

    parameters = custom_preprocessor(device, state_dict)
    box_head = TtCTResNetHead(parameters=parameters, device=device)

    input = torch.rand((1, 64, 112, 168))
    torch_output = reference.forward(input)

    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)

    tt_output = box_head.forward(tt_input)

    assert_with_pcc(ttnn.to_torch(tt_output[0]), torch_output[0], 0.99)
    assert_with_pcc(ttnn.to_torch(tt_output[1]), torch_output[1], 0.99)
    assert_with_pcc(ttnn.to_torch(tt_output[2]), torch_output[2], 0.989)
