# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet
from models.experimental.transfuser.tt.lidar_center_net import LidarCenterNet as TtLidarCenterNet
from models.experimental.transfuser.tests.test_gpt import create_gpt_preprocessor

from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


def create_lidar_center_net_head_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        # Process each head's parameters
        for head_name in [
            "heatmap_head",
            "wh_head",
            "offset_head",
            "yaw_class_head",
            "yaw_res_head",
            "velocity_head",
            "brake_head",
        ]:
            if hasattr(torch_model, head_name):
                head = getattr(torch_model, head_name)

                # Get output channels for this head
                out_channels = head[2].weight.shape[0]  # From second conv layer

                # Note: We cannot use prepare_conv_weights here because we need
                # the full conv2d parameters (batch_size, input_height, etc.)
                # which are only available at runtime, not during preprocessing.
                # So we keep weights in PyTorch format and convert at runtime.
                parameters[head_name] = {}

                # Store weights in PyTorch format - will be prepared during first forward pass
                parameters[head_name]["conv1_weight"] = ttnn.from_torch(
                    head[0].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv1_bias"] = ttnn.from_torch(
                    head[0].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

                parameters[head_name]["conv2_weight"] = ttnn.from_torch(
                    head[2].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_bias"] = ttnn.from_torch(
                    head[2].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

        return parameters

    return custom_preprocessor


def get_mesh_mappers(device):
    if device.get_num_devices() != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, target_point_image_shape, img_shape, lidar_bev_shape",
    [
        ("regnety_032", "regnety_032", 4, False, (1, 1, 256, 256), (1, 3, 160, 704), (1, 2, 256, 256))
    ],  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_lidar_center_net(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    target_point_image_shape,
    img_shape,
    lidar_bev_shape,
    input_dtype,
    weight_dtype,
):
    image = torch.randn(img_shape)
    lidar_bev = torch.randn(lidar_bev_shape)
    target_point = torch.randn(1, 2)
    target_point_image = torch.randn(target_point_image_shape)
    velocity = torch.randn(1, 1)

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    # setting machine to avoid loading files
    config = GlobalConfig(setting="eval")
    config.n_layer = n_layer
    config.use_target_point_image = True

    ref_layer = LidarCenterNet(
        config,
        backbone="transFuser",
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=use_velocity,
    ).eval()

    # pred_wp, rotated_bboxes = ref_layer.forward_ego(image, lidar_bev, target_point, target_point_image, velocity)
    torch_features, torch_fused = ref_layer.forward_ego(image, lidar_bev, target_point, target_point_image, velocity)

    torch_model = ref_layer._model

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    gpt1_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer1,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer1"] = gpt1_parameters
    gpt2_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer2,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer2"] = gpt2_parameters
    gpt3_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer3,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer3"] = gpt3_parameters
    gpt4_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer4,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer4"] = gpt4_parameters

    # Preprocess model parameters
    parameters["head"] = preprocess_model_parameters(
        initialize_model=lambda: ref_layer.head,
        custom_preprocessor=create_lidar_center_net_head_preprocessor(device, weight_dtype),
        device=device,
    )

    tt_layer = TtLidarCenterNet(
        device,
        parameters,
        config,
        backbone="transFuser",
    )

    features, fused_features = tt_layer.forward_ego(image, lidar_bev, target_point, target_point_image, velocity)

    tt_features_torch = []
    fpn_names = ["p2", "p3", "p4", "p5"]
    for i, (feature, name) in enumerate(zip(features, fpn_names)):
        tt_feat = ttnn.to_torch(
            feature,
            device=device,
            mesh_composer=output_mesh_composer,
        )

        # Permute NHWC -> NCHW
        tt_feat = tt_feat.permute(0, 3, 1, 2)
        tt_features_torch.append(tt_feat)

    # Validate output_fused_tensor
    tt_fused_torch = ttnn.to_torch(
        fused_features,
        device=device,
        mesh_composer=output_mesh_composer,
    )

    # Deallocate output tensors
    for feature in features:
        ttnn.deallocate(feature)
    ttnn.deallocate(fused_features)

    # Validate FPN features
    fpn_pcc_results = []
    for torch_feat, tt_feat, name in zip(torch_features, tt_features_torch, fpn_names):
        pcc_passed, pcc_msg = check_with_pcc(torch_feat, tt_feat, pcc=0.90)
        fpn_pcc_results.append((pcc_passed, pcc_msg))
        logger.info(f"{name} PCC: {pcc_msg}")

    # Validate fused features
    fused_pcc_passed, fused_pcc_msg = check_with_pcc(torch_fused, tt_fused_torch, pcc=0.90)
    logger.info(f"Fused Features PCC: {fused_pcc_msg}")

    # All outputs must pass
    all_fpn_passed = all(result[0] for result in fpn_pcc_results)
    overall_passed = all_fpn_passed and fused_pcc_passed

    assert overall_passed, logger.error(f"PCC check failed - FPN: {fpn_pcc_results}, Fused: {fused_pcc_msg}")

    return overall_passed, f"FPN: {fpn_pcc_results}, Fused: {fused_pcc_msg}"
