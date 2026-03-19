# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_common import (
    preprocess_linear_bias,
    preprocess_linear_weight,
    tt_mlp_block,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_mlp_block(device, reset_seeds):
    """Test MLP block (Linear -> GELU -> Linear) against PyTorch reference."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 64
    in_features = 128
    hidden_features = 256
    out_features = 128

    # Build reference PyTorch MLP
    linear1 = torch.nn.Linear(in_features, hidden_features)
    linear2 = torch.nn.Linear(hidden_features, out_features)
    torch_mlp = torch.nn.Sequential(linear1, torch.nn.GELU(), linear2)
    torch_mlp = torch_mlp.to(torch.bfloat16).eval()

    # Reference input and output
    torch_input = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_mlp(torch_input)

    # Preprocess weights and biases for ttnn
    weight1 = preprocess_linear_weight(linear1.weight.detach().to(torch.bfloat16))
    bias1 = preprocess_linear_bias(linear1.bias.detach().to(torch.bfloat16))
    weight2 = preprocess_linear_weight(linear2.weight.detach().to(torch.bfloat16))
    bias2 = preprocess_linear_bias(linear2.bias.detach().to(torch.bfloat16))

    # Move weights to device
    weight1 = ttnn.to_device(weight1, device)
    bias1 = ttnn.to_device(bias1, device)
    weight2 = ttnn.to_device(weight2, device)
    bias2 = ttnn.to_device(bias2, device)

    # Move input to device
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run ttnn MLP block
    tt_output = tt_mlp_block(tt_input, weight1, bias1, weight2, bias2, activation="gelu")

    # Convert back to torch for comparison
    output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, output, 0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_preprocess_linear_weight(device, reset_seeds):
    """Test that preprocess_linear_weight correctly transposes the weight tensor."""
    torch.manual_seed(0)

    in_features = 64
    out_features = 128

    # PyTorch linear weight shape is [out_features, in_features]
    torch_weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    tt_weight = preprocess_linear_weight(torch_weight)

    # After transposition, shape should be [in_features, out_features]
    assert tt_weight.shape[0] == in_features, (
        f"Expected first dim {in_features}, got {tt_weight.shape[0]}"
    )
    assert tt_weight.shape[1] == out_features, (
        f"Expected second dim {out_features}, got {tt_weight.shape[1]}"
    )

    # Verify values are correctly transposed
    tt_weight_torch = ttnn.to_torch(tt_weight)
    expected = torch_weight.T
    assert torch.allclose(tt_weight_torch.to(torch.float32), expected.to(torch.float32)), (
        "Transposed weight values do not match"
    )
