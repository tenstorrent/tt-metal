# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests self-attention with TTNN acceleration."""

import pytest
import torch

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.attention import SelfAttention, SelfAttentionConfig, TTNNSelfAttention
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_self_attention(device):
    """Test SELF Attention with TTNN acceleration."""
    config = SelfAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
    )
    model = SelfAttention(config).to(dtype=torch.bfloat16)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    inputs = TorchTTNNTensor(torch.randn((1, 5, 768), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNSelfAttention.from_torch(model)
    set_device(ttnn_model, device)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "SelfAttention")
