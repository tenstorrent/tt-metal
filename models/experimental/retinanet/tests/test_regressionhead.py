# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import pickle
import os
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.retinanet.TTNN.regression_head import ttnn_retinanet_regression_head


def create_regression_head_parameters(torch_model, device):
    """Convert PyTorch regression head weights to TTNN format."""
    parameters = {}

    # Define grid configuration
    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Convert 4 conv layers (Conv2d + GroupNorm weights)
    parameters["conv"] = []
    for i in range(4):
        # Conv2d weights
        conv_weight = torch_model.conv[i][0].weight.detach().to(torch.bfloat16)

        # GroupNorm weights - MUST use create_group_norm_weight_bias_rm()
        norm_weight = torch_model.conv[i][1].weight.detach()
        norm_bias = torch_model.conv[i][1].bias.detach()

        # Format GroupNorm parameters using helper function
        formatted_norm_weight = ttnn.create_group_norm_weight_bias_rm(
            norm_weight, num_channels=256, num_cores_x=grid_size.y
        )
        formatted_norm_bias = ttnn.create_group_norm_weight_bias_rm(
            norm_bias, num_channels=256, num_cores_x=grid_size.y
        )

        conv_params = {
            "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, device=device),
            "norm_weight": ttnn.from_torch(
                formatted_norm_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "norm_bias": ttnn.from_torch(
                formatted_norm_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        }

        parameters["conv"].append(conv_params)

    # Convert bbox_reg layer
    bbox_weight = torch_model.bbox_reg.weight.detach().to(torch.bfloat16)
    bbox_bias = torch_model.bbox_reg.bias.detach().to(torch.bfloat16)

    parameters["bbox_reg"] = {
        "weight": ttnn.from_torch(bbox_weight, dtype=ttnn.bfloat16, device=device),
        "bias": ttnn.from_torch(bbox_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, device=device),
    }

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_retinanet_v2_regression_head_ttnn_5_fpn_with_real_features(device, pcc, reset_seeds):
    """Test TTNN RetinaNet V2 regression head with 5 FPN levels using real features."""
    torch.manual_seed(0)

    # Load pretrained model
    torch_model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    torch_model.eval()
    torch_model = torch_model.to(dtype=torch.bfloat16)
    regression_head = torch_model.head.regression_head

    # Load pickled FPN features
    pickle_path = "models/experimental/retinanet/data/fpn_features.pkl"

    if os.path.exists(pickle_path):
        print(f"✓ Loading real FPN features from {pickle_path}")
        with open(pickle_path, "rb") as f:
            saved_data = pickle.load(f)

        torch_features = saved_data["features"]
        input_shapes = saved_data["input_shapes"]
        batch_size = saved_data["batch_size"]
        in_channels = saved_data["in_channels"]

        print(f"  Loaded {len(torch_features)} FPN levels:")
        for i, feat in enumerate(torch_features):
            print(f"    Level {i}: {feat.shape}")
    else:
        print(f"Pickle file not found at {pickle_path}, using random features")
        print(f"  Run 'python models/experimental/retinanet/tests/generate_fpn_features.py' first")

        # Fallback to random features
        batch_size = 1
        in_channels = 256
        input_shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]

        torch_features = [torch.randn(batch_size, in_channels, H, W, dtype=torch.bfloat16) for H, W in input_shapes]

    num_anchors = 9

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = regression_head(torch_features)

    # Convert to TTNN (NHWC format) - convert all features to device
    ttnn_features = [
        ttnn.from_torch(
            feature.permute(0, 2, 3, 1),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for feature in torch_features
    ]

    # Create TTNN parameters
    ttnn_parameters = create_regression_head_parameters(regression_head, device)

    # TTNN forward pass
    ttnn_output = ttnn_retinanet_regression_head(
        feature_maps=ttnn_features,
        parameters=ttnn_parameters,
        device=device,
        in_channels=in_channels,
        num_anchors=num_anchors,
        batch_size=batch_size,
        input_shapes=input_shapes,
    )

    # Convert back to PyTorch for comparison
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Assert PCC
    passed, pcc_msg = assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)
    assert passed, f"PCC test failed: {pcc_msg}"

    print(f"✓ TTNN regression head test passed with 5 FPN levels! {pcc_msg}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"TTNN output shape: {ttnn_output_torch.shape}")
