# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for ViT model with TTNN backend."""

import pytest
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

from models.experimental.tt_symbiote.modules.conv import TTNNBottleneck
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_resnet(device):
    """Test Resnet model with TTNN acceleration."""

    model = resnet50(pretrained=True).to(torch.bfloat16)
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        Bottleneck: TTNNBottleneck,
    }
    register_module_replacement_dict(model, nn_to_ttnn, model_config={"program_config_ffn": {}})
    set_device(model, device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    result = model(torch.randn(1, 3, 224, 224, dtype=torch.bfloat16))
    print(result)
