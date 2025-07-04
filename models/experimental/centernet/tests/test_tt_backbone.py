# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_resnet import TtBasicBlock, TtResNet
from models.experimental.centernet.reference.resnet import BasicBlock, ResNet
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet_backbone(device, reset_seeds):
    with torch.no_grad():
        reference = ResNet(BasicBlock, [2, 2, 2, 2])
        reference.eval()

    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if "backbone" in k:
            new_state_dict[k] = v

    torch_state_dict = {}
    for k, v in state_dict.items():
        if "backbone" in k:
            torch_state_dict[k[9:]] = v
    reference.load_state_dict(torch_state_dict)

    parameters = custom_preprocessor(device, state_dict)
    backbone = TtResNet(TtBasicBlock, [2, 2, 2, 2], parameters=parameters, base_address="backbone", device=device)

    input = torch.rand((1, 3, 448, 672))
    torch_ouput = reference.forward(input)

    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    tt_ouput = backbone.forward(tt_input)

    assert_with_pcc(ttnn.to_torch(tt_ouput[0]), torch_ouput[0], 0.99)
    assert_with_pcc(ttnn.to_torch(tt_ouput[1]), torch_ouput[1], 0.99)
    assert_with_pcc(ttnn.to_torch(tt_ouput[2]), torch_ouput[2], 0.99)
    assert_with_pcc(ttnn.to_torch(tt_ouput[3]), torch_ouput[3], 0.99)
