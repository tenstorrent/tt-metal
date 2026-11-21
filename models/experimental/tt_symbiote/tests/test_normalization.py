# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import ttnn
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_ttnn_layer_norm_uses_module_eps(device, eps):
    torch.manual_seed(0)
    hidden = 768
    torch_layer = nn.LayerNorm(hidden, eps=eps)
    with torch.no_grad():
        torch_layer.weight.copy_(torch.randn(hidden) * 0.5 + 1.0)
        torch_layer.bias.copy_(torch.randn(hidden) * 0.1)

    module = TTNNLayerNorm.from_torch(torch_layer)
    module.to_device(device)
    module.preprocess_weights()
    module.move_weights_to_device()

    # Low-variance rows: the result depends on eps, so the module must use the torch layer's eps.
    x = (torch.randn(1, 1, 64, hidden) * 0.005).bfloat16()
    output = module.forward(ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device))

    expected = torch.nn.functional.layer_norm(
        x.float(), (hidden,), torch_layer.weight.bfloat16().float(), torch_layer.bias.bfloat16().float(), eps
    )
    max_error = (ttnn.to_torch(output).float() - expected).abs().max().item()
    assert max_error < 0.1, f"max abs error {max_error}"
