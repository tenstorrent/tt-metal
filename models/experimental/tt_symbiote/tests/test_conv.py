# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test NHWC Conv"""

import pytest
import torch
from torch import nn

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.conv import (
    NHWCConvBNActivationPytorch,
    NHWCConvBNPytorch,
    NHWCConvPytorch,
    TTNNConv2dBNActivationNHWC,
    TTNNConv2dBNNHWC,
    TTNNConv2dNHWC,
)
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_conv(device):
    """Test NHWC Conv with TTNN acceleration."""

    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1).to(dtype=torch.bfloat16)
    model = NHWCConvPytorch(conv)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    inputs = TorchTTNNTensor(torch.randn((1, 224, 224, 3), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNConv2dNHWC.from_torch(conv)
    set_device(ttnn_model, device)
    outputs_ttnn = ttnn_model(inputs)
    outputs_ttnn.elem = None  # Force using TTNN tensor only
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Conv2dNHWC")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_conv_bn(device):
    """Test NHWC Conv with TTNN acceleration."""

    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1).to(dtype=torch.bfloat16)
    bn = nn.BatchNorm2d(num_features=16).to(dtype=torch.bfloat16)
    model = NHWCConvBNPytorch(conv, bn)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    inputs = TorchTTNNTensor(torch.randn((1, 224, 224, 3), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNConv2dBNNHWC.from_torch(conv, bn)
    set_device(ttnn_model, device)
    outputs_ttnn = ttnn_model(inputs)
    outputs_ttnn.elem = None  # Force using TTNN tensor only
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Conv2dBNNHWC")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_conv_bn_relu(device):
    """Test NHWC Conv with TTNN acceleration."""

    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1).to(dtype=torch.bfloat16)
    bn = nn.BatchNorm2d(num_features=16).to(dtype=torch.bfloat16)
    relu = nn.ReLU()
    model = NHWCConvBNActivationPytorch(conv, bn, relu)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    inputs = TorchTTNNTensor(torch.randn((1, 224, 224, 3), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNConv2dBNActivationNHWC.from_torch(conv, bn, relu)
    set_device(ttnn_model, device)
    outputs_ttnn = ttnn_model(inputs)
    outputs_ttnn.elem = None  # Force using TTNN tensor only
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Conv2dBNActivationNHWC")
