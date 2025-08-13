# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.experimental.panoptic_deeplab.tt.tt_pytorch_aspp import ASPP
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.panoptic_deeplab.tt.tt_aspp import TtASPP


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, input_height, input_width, dilations, norm, activation, dropout, pool_kernel_size",
    [
        (1, 2048, 256, 32, 64, [6, 12, 18], "ln", "relu", 0.0, (32, 64)),
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
    pool_kernel_size,
):
    torch.manual_seed(0)
    shared_weight_tensor_kernel1 = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    shared_weight_tensor_kernel3 = torch.randn(out_channels, in_channels, 3, 3, dtype=torch.bfloat16)
    shared_weight_tensor_kernel1_output5 = torch.randn(out_channels, 5 * out_channels, 1, 1, dtype=torch.bfloat16)

    torch_input = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16)

    torch_model = ASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        dilations=dilations,
        norm="LN" if norm == "ln" else "",
        activation=torch.nn.ReLU() if activation == "relu" else torch.nn.SiLU(),
        pool_kernel_size=pool_kernel_size,
        dropout=dropout,
        shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
        shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
        shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
    )
    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()
    torch_output = torch_model(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    ttnn_model = TtASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        dilations=dilations,
        device=device,
        norm=norm,
        activation=activation,
        dropout=dropout,
        pool_kernel_size=pool_kernel_size,
        shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
        shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
        shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
    )

    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output, pcc=0.95)

    print(f"PCC: {pcc_message}")
    assert pcc_passed, f"PCC check failed: {pcc_message}"
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

    nchw = (batch_size, in_channels, input_height, input_width)

    torch_input = torch.randn(nchw, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create and run model
    ttnn_model = TtASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        dilations=dilations,
        device=device,
        norm="ln",
        activation="relu",
        dropout=0.5,
        pool_kernel_size=(4, 4),
    )

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, (0, 3, 1, 2))

    # Basic checks
    assert ttnn_output is not None
    assert ttnn_output.shape[0] == batch_size  # Batch dimension
    assert ttnn_output.shape[3] == out_channels  # Channel dimension (NHWC)
    assert ttnn_output.shape[1] == input_height  # Height dimension
    assert ttnn_output.shape[2] == input_width  # Width dimension
