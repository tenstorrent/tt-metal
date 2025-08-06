# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.panoptic_deeplab.tt.tt_aspp import TtASPP


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, input_height, input_width, dilations, norm, activation, dropout",
    [
        # Basic test cases
        (1, 256, 256, 32, 32, [6, 12, 18], "gn", "relu", 0.0),
        (1, 512, 256, 16, 16, [6, 12, 18], "gn", "silu", 0.1),
        (2, 256, 128, 64, 64, [3, 6, 9], "", "relu", 0.0),  # No norm case
        # Different dilation patterns
        (1, 128, 128, 32, 32, [2, 4, 8], "gn", "relu", 0.0),
        (1, 256, 256, 48, 48, [1, 2, 4], "gn", "silu", 0.0),
        # Larger input sizes
        (1, 256, 256, 128, 128, [6, 12, 18], "gn", "relu", 0.0),
    ],
)
def test_ttnn_aspp(
    device,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    dilations,
    norm,
    activation,
    dropout,
):
    torch.manual_seed(0)

    # Create PyTorch reference implementation (simplified ASPP)
    class PyTorchASPP(torch.nn.Module):
        def __init__(self, in_channels, out_channels, dilations, norm, activation, dropout):
            super().__init__()
            self.dropout = dropout
            use_bias = norm == ""

            # 1x1 conv
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, 1, bias=use_bias)

            # Dilated convs
            self.dilated_convs = torch.nn.ModuleList(
                [torch.nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=use_bias) for d in dilations]
            )

            # Global avg pool + 1x1 conv
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.pool_conv = torch.nn.Conv2d(in_channels, out_channels, 1, bias=True)

            # Projection
            self.project = torch.nn.Conv2d(5 * out_channels, out_channels, 1, bias=use_bias)

            # Activation
            if activation == "relu":
                self.activation = torch.nn.ReLU()
            elif activation == "silu":
                self.activation = torch.nn.SiLU()

        def forward(self, x):
            size = x.shape[-2:]
            res = []

            # 1x1 conv
            res.append(self.activation(self.conv1x1(x)))

            # Dilated convs
            for conv in self.dilated_convs:
                res.append(self.activation(conv(x)))

            # Global pooling branch
            pooled = self.global_pool(x)
            pooled = self.activation(self.pool_conv(pooled))
            pooled = torch.nn.functional.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
            res.append(pooled)

            # Concatenate and project
            res = torch.cat(res, dim=1)
            res = self.activation(self.project(res))

            if self.dropout > 0:
                res = torch.nn.functional.dropout(res, p=self.dropout, training=self.training)

            return res

    # Create input tensor
    torch_input = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16)

    # PyTorch reference
    torch_model = PyTorchASPP(in_channels, out_channels, dilations, norm, activation, dropout)
    torch_model.eval()
    torch_output = torch_model(torch_input)

    # Convert to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create TTNN model
    ttnn_model = TtASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        dilations=dilations,
        device=device,
        norm=norm,
        activation=activation,
        dropout=dropout,
    )

    # Run TTNN model
    ttnn_output = ttnn_model(ttnn_input)

    # Convert back to PyTorch format (NCHW)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    # Compare outputs
    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output, pcc=0.95)
    print(f"PCC: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"

    # Basic shape check
    assert torch_output.shape == ttnn_output.shape, f"Shape mismatch: {torch_output.shape} vs {ttnn_output.shape}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_aspp_basic_functionality(device):
    """Basic smoke test to ensure the module can be instantiated and run"""
    torch.manual_seed(0)

    # Simple test case
    batch_size, in_channels, out_channels = 1, 256, 256
    input_height, input_width = 32, 32
    dilations = [6, 12, 18]

    # Create input
    torch_input = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create and run model
    ttnn_model = TtASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        dilations=dilations,
        device=device,
        norm="gn",
        activation="relu",
        dropout=0.0,
    )

    ttnn_output = ttnn_model(ttnn_input)

    # Basic checks
    assert ttnn_output is not None
    assert ttnn_output.shape[0] == batch_size  # Batch dimension
    assert ttnn_output.shape[3] == out_channels  # Channel dimension (NHWC)
    assert ttnn_output.shape[1] == input_height  # Height dimension
    assert ttnn_output.shape[2] == input_width  # Width dimension
