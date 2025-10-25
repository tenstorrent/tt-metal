# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
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
            if head_name == "heatmap_head":
                weight_dtype = ttnn.float32
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
        ("regnety_032", "regnety_032", 4, False, (1, 1, 256, 256), (1, 3, 160, 704), (1, 2, 256, 256)),
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
    torch.manual_seed(8)
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
    ref_outputs, pred_wp, rotated_bboxes, results = ref_layer.forward_ego(
        image, lidar_bev, target_point, target_point_image, velocity
    )

    # Unpack list outputs (each contains one tensor since we have single scale)
    (
        ref_center_heatmap_list,
        ref_wh_list,
        ref_offset_list,
        ref_yaw_class_list,
        ref_yaw_res_list,
        ref_velocity_list,
        ref_brake_list,
    ) = ref_outputs

    # Extract single tensors from lists
    ref_center_heatmap = ref_center_heatmap_list[0]
    ref_wh = ref_wh_list[0]
    ref_offset = ref_offset_list[0]
    ref_yaw_class = ref_yaw_class_list[0]
    ref_yaw_res = ref_yaw_res_list[0]
    ref_velocity = ref_velocity_list[0]
    ref_brake = ref_brake_list[0]
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

    tt_outputs, tt_pred_wp = tt_layer.forward_ego(image, lidar_bev, target_point, target_point_image, velocity)

    # Unpack list outputs
    (
        tt_center_heatmap_list,
        tt_wh_list,
        tt_offset_list,
        tt_yaw_class_list,
        tt_yaw_res_list,
        tt_velocity_list,
        tt_brake_list,
    ) = tt_outputs

    # Extract single tensors from lists
    tt_center_heatmap = tt_center_heatmap_list[0]
    tt_wh = tt_wh_list[0]
    tt_offset = tt_offset_list[0]
    tt_yaw_class = tt_yaw_class_list[0]
    tt_yaw_res = tt_yaw_res_list[0]
    tt_velocity = tt_velocity_list[0]
    tt_brake = tt_brake_list[0]

    # Convert TTNN outputs back to torch (NHWC -> NCHW)
    tt_center_heatmap_torch = tt2torch_tensor(tt_center_heatmap).permute(0, 3, 1, 2)
    tt_center_heatmap_torch = tt_center_heatmap_torch.reshape(ref_center_heatmap.shape)
    tt_wh_torch = tt2torch_tensor(tt_wh).permute(0, 3, 1, 2)
    tt_wh_torch = tt_wh_torch.reshape(ref_wh.shape)
    tt_offset_torch = tt2torch_tensor(tt_offset).permute(0, 3, 1, 2)
    tt_offset_torch = tt_offset_torch.reshape(ref_offset.shape)
    tt_yaw_class_torch = tt2torch_tensor(tt_yaw_class).permute(0, 3, 1, 2)
    tt_yaw_class_torch = tt_yaw_class_torch.reshape(ref_yaw_class.shape)
    tt_yaw_res_torch = tt2torch_tensor(tt_yaw_res).permute(0, 3, 1, 2)
    tt_yaw_res_torch = tt_yaw_res_torch.reshape(ref_yaw_res.shape)
    tt_velocity_torch = tt2torch_tensor(tt_velocity).permute(0, 3, 1, 2)
    tt_velocity_torch = tt_velocity_torch.reshape(ref_velocity.shape)
    tt_brake_torch = tt2torch_tensor(tt_brake).permute(0, 3, 1, 2)
    tt_brake_torch = tt_brake_torch.reshape(ref_brake.shape)

    # # Dump ref_center_heatmap to pickle file
    # with open('ref_center_heatmap.pkl', 'wb') as f:
    #     pickle.dump(ref_center_heatmap, f)
    # logger.info("Dumped ref_center_heatmap to ref_center_heatmap.pkl")

    # ttnn_function = ttnn.sigmoid
    # golden_function = ttnn.get_golden_function(ttnn_function)
    # tt_center_heatmap_torch = golden_function(tt_center_heatmap_torch, device=device)
    # Validate center heatmap

    does_pass, heatmap_pcc_message = check_with_pcc(ref_center_heatmap, tt_center_heatmap_torch, 0.90)
    logger.info(f"Center Heatmap PCC: {heatmap_pcc_message}")
    assert does_pass, f"Center Heatmap PCC check failed: {heatmap_pcc_message}"

    # Validate WH prediction
    does_pass, wh_pcc_message = check_with_pcc(ref_wh, tt_wh_torch, 0.90)
    logger.info(f"WH PCC: {wh_pcc_message}")
    assert does_pass, f"WH PCC check failed: {wh_pcc_message}"

    # Validate offset prediction
    does_pass, offset_pcc_message = check_with_pcc(ref_offset, tt_offset_torch, 0.90)
    logger.info(f"Offset PCC: {offset_pcc_message}")
    assert does_pass, f"Offset PCC check failed: {offset_pcc_message}"

    # Validate yaw class prediction
    does_pass, yaw_class_pcc_message = check_with_pcc(ref_yaw_class, tt_yaw_class_torch, 0.90)
    logger.info(f"Yaw Class PCC: {yaw_class_pcc_message}")
    assert does_pass, f"Yaw Class PCC check failed: {yaw_class_pcc_message}"

    # Validate yaw residual prediction
    does_pass, yaw_res_pcc_message = check_with_pcc(ref_yaw_res, tt_yaw_res_torch, 0.90)
    logger.info(f"Yaw Residual PCC: {yaw_res_pcc_message}")
    assert does_pass, f"Yaw Residual PCC check failed: {yaw_res_pcc_message}"

    # Validate velocity prediction
    does_pass, velocity_pcc_message = check_with_pcc(ref_velocity, tt_velocity_torch, 0.90)
    logger.info(f"Velocity PCC: {velocity_pcc_message}")
    assert does_pass, f"Velocity PCC check failed: {velocity_pcc_message}"

    # Validate brake prediction
    does_pass, brake_pcc_message = check_with_pcc(ref_brake, tt_brake_torch, 0.90)
    logger.info(f"Brake PCC: {brake_pcc_message}")
    assert does_pass, f"Brake PCC check failed: {brake_pcc_message}"

    does_pass, pred_wp_pcc_message = check_with_pcc(pred_wp, tt_pred_wp, 0.90)
    logger.info(f"pred wp PCC: {pred_wp_pcc_message}")
    assert does_pass, f"pred wp PCC check failed: {pred_wp_pcc_message}"

    # After the pred_wp PCC check, add bbox post-processing for TTNN outputs

    # Convert TTNN outputs to torch for get_bboxes (it expects torch tensors)
    tt_preds_torch = (
        [tt_center_heatmap_torch],
        [tt_wh_torch],
        [tt_offset_torch],
        [tt_yaw_class_torch],
        [tt_yaw_res_torch],
        [tt_velocity_torch],
        [tt_brake_torch],
    )

    # Call get_bboxes on the reference head (reusing the same logic)
    tt_results = ref_layer.head.get_bboxes(*tt_preds_torch)
    does_pass, box_pcc_message = check_with_pcc(results[0][0], tt_results[0][0], 0.90)
    logger.info(f"box PCC: {box_pcc_message}")
    assert does_pass, f"box PCC check failed: {box_pcc_message}"
    tt_bboxes, _ = tt_results[0]

    # Filter by confidence threshold
    tt_bboxes = tt_bboxes[tt_bboxes[:, -1] > config.bb_confidence_threshold]

    # Convert to metric coordinates
    tt_rotated_bboxes = []
    for bbox in tt_bboxes.detach().cpu().numpy():
        bbox_metric = ref_layer.get_bbox_local_metric(bbox)
        tt_rotated_bboxes.append(bbox_metric)

    # Compare bbox counts
    logger.info(f"Reference bboxes count: {len(rotated_bboxes)}")
    logger.info(f"TTNN bboxes count: {len(tt_rotated_bboxes)}")

    if does_pass:
        logger.info("LidarCenterNet Passed!")
    else:
        logger.warning("LidarCenterNet Failed!")
