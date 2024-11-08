# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from torchvision import models
from models.demos.squeezenet.reference.squeezenet import squeezenet
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_reference_model(device):
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    torch_squeezenet.eval()
    state_dict = torch_squeezenet.state_dict()
    torch_in = torch.randn(1, 3, 224, 224)
    torch_out = torch_squeezenet(torch_in)
    ref_squeezenet_out = squeezenet(state_dict=state_dict, input=torch_in)
    assert_with_pcc(torch_out, ref_squeezenet_out, pcc=0.99)
