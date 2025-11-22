# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import pickle
import os
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.retinanet.tt.tt_regression_head import ttnn_retinanet_regression_head


def create_regression_head_parameters(torch_head, device, model_config):
    """Convert PyTorch regression head weights to TTNN format."""
    parameters = {}

    # Define grid configuration
    # Grid size for GroupNorm
    grid_size = ttnn.CoreGrid(y=8, x=8)
    layout = (
        ttnn.TILE_LAYOUT if model_config["WEIGHTS_DTYPE"] in [ttnn.bfloat8_b, ttnn.bfloat4_b] else ttnn.ROW_MAJOR_LAYOUT
    )
    # Convert 4 conv layers (Conv2d + GroupNorm weights)
    parameters["conv"] = []
    for i in range(4):
        # Conv2d weights
        # conv_weight = torch_head.conv[i][0].weight.detach().to(torch.bfloat16)  # Was: torch.bfloat16
        # bias=torch.zeros(conv_weight.shape[0])
        conv_weight = torch_head.conv[i][0].weight.detach().to(torch.bfloat16)
        bias = torch.zeros(conv_weight.shape[0]).to(torch.bfloat16)

        # GroupNorm weights - MUST use create_group_norm_weight_bias_rm()
        norm_weight = torch_head.conv[i][1].weight.detach()
        norm_bias = torch_head.conv[i][1].bias.detach()

        # Format GroupNorm parameters using helper function
        formatted_norm_weight = ttnn.create_group_norm_weight_bias_rm(
            norm_weight, num_channels=256, num_cores_x=grid_size.y
        )
        formatted_norm_bias = ttnn.create_group_norm_weight_bias_rm(
            norm_bias, num_channels=256, num_cores_x=grid_size.y
        )
        # Prepare weights using ttnn.prepare_conv_weights
        prepared_weight = ttnn.prepare_conv_weights(
            weight_tensor=ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16),
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=conv_weight.shape[1],
            out_channels=conv_weight.shape[0],
            batch_size=1,
            input_height=64,  # Adjust based on FPN level
            input_width=64,  # Adjust based on FPN level
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            has_bias=True,
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
        )

        # Prepare bias using ttnn.prepare_conv_bias
        prepared_bias = ttnn.prepare_conv_bias(
            bias_tensor=ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=conv_weight.shape[1],
            out_channels=conv_weight.shape[0],
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
            conv_config=ttnn.Conv2dConfig(weights_dtype=model_config["WEIGHTS_DTYPE"]),
        )
        conv_params = {
            "weight": prepared_weight,
            "bias": prepared_bias,
            "norm_weight": ttnn.from_torch(
                formatted_norm_weight,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "norm_bias": ttnn.from_torch(
                formatted_norm_bias,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        }

        parameters["conv"].append(conv_params)

        # Convert bbox_reg layer
        bbox_weight = torch_head.bbox_reg.weight.detach().to(torch.bfloat16)
        bbox_bias = torch_head.bbox_reg.bias.detach().to(torch.bfloat16)
        # Convert to TTNN tensor (host)
        bbox_weight_ttnn = ttnn.from_torch(bbox_weight, dtype=model_config["WEIGHTS_DTYPE"])
        # First convert to TTNN format
        bbox_bias_ttnn = ttnn.from_torch(
            bbox_bias.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
        )
        # Use prepare_conv_weights to transform to proper format
        prepared_bbox_weight = ttnn.prepare_conv_weights(
            weight_tensor=bbox_weight_ttnn,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Match your input config
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=256,  # From FPN output
            out_channels=bbox_weight.shape[0],  # num_anchors * 4
            batch_size=1,
            input_height=64,  # Use largest FPN size for preparation
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            has_bias=True,
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
            output_dtype=model_config["WEIGHTS_DTYPE"],
            conv_config=ttnn.Conv2dConfig(weights_dtype=model_config["WEIGHTS_DTYPE"]),
            compute_config=None,
        )
        # Prepare the bias using prepare_conv_bias
        prepared_bbox_bias = ttnn.prepare_conv_bias(
            bias_tensor=bbox_bias_ttnn,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=256,
            out_channels=bbox_weight.shape[0],
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
            conv_config=ttnn.Conv2dConfig(weights_dtype=model_config["WEIGHTS_DTYPE"]),  # Must have weights_dtype set
        )
        parameters["bbox_reg"] = {
            "weight": prepared_bbox_weight,
            "bias": prepared_bbox_bias,
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
    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }
    # Create TTNN parameters
    ttnn_parameters = create_regression_head_parameters(regression_head, device, model_config)

    # TTNN forward pass
    ttnn_output = ttnn_retinanet_regression_head(
        feature_maps=ttnn_features,
        parameters=ttnn_parameters,
        device=device,
        in_channels=in_channels,
        num_anchors=num_anchors,
        batch_size=batch_size,
        input_shapes=input_shapes,
        model_config=model_config,
        optimization_profile="optimized",
    )

    # Convert back to PyTorch for comparison
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Assert PCC
    passed, pcc_msg = assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)
    assert passed, f"PCC test failed: {pcc_msg}"

    print(f"✓ TTNN regression head test passed with 5 FPN levels! {pcc_msg}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"TTNN output shape: {ttnn_output_torch.shape}")
