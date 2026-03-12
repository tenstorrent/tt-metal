# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.synthesizer.modules import ResidualCouplingLayer as TorchResidualCouplingLayer
from models.demos.rvc.tt_impl.synthesizer.modules import ResidualCouplingLayer as TTResidualCouplingLayer
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_residual_coupling_layer(device):
    torch.manual_seed(0)

    batch_size = 1
    channels = 16
    hidden_channels = 16
    gin_channels = 8
    input_length = 64
    kernel_size = 3
    dilation_rate = 1
    n_layers = 2

    torch_layer = TorchResidualCouplingLayer(
        channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
    ).eval()

    torch_x = torch.randn(batch_size, channels, input_length, dtype=torch.float32)
    torch_g = torch.randn(batch_size, gin_channels, input_length, dtype=torch.float32)
    torch_output = torch_layer(torch_x, g=torch_g)

    tt_layer = TTResidualCouplingLayer(
        device=device,
        channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
    )

    parameters = {f"flow.{k}": v for k, v in torch_layer.state_dict().items()}
    tt_layer.load_parameters(parameters=parameters, prefix="flow.")

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_g = ttnn.from_torch(
        torch_g.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = tt_layer(tt_x, g=tt_g)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.permute(0, 2, 1)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.98)
