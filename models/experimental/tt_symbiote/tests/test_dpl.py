# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test DPL with TTNN backend."""

import os

import torch
from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * 2.0
        x = x + 3.0
        return x


def test_dpl(device):
    """Test DPL model with TTNN acceleration."""
    model = CustomModule()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    assert (
        os.environ.get("TT_SYMBIOTE_RUN_MODE") == "DPL"
    ), f"Expected TT_SYMBIOTE_RUN_MODE environment variable to be 'DPL', got {os.environ.get('TT_SYMBIOTE_RUN_MODE')}"
    inputs = TorchTTNNTensor(torch.tensor([1.0, 2.0, 3.0]))
    inputs.ttnn_tensor = ttnn.to_device(inputs.to_ttnn, device)
    outputs = model(inputs)
    assert (outputs.elem == outputs.to_torch).all()
    result = outputs.elem.clone()
    outputs.elem = None  # Force using TTNN tensor only
    ttnn_result = outputs.to_torch
    assert (result == ttnn_result).all()
