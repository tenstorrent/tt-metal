# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
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


def compare_boxes_pcc(ref_boxes, torch_boxes):
    """
    Compare all reference boxes with all torch boxes using PCC.
    Returns the top len(ref_boxes) PCC scores with their indices.
    """
    pcc_scores = []

    print("Computing PCC between all pairs of boxes...")

    # Compare each reference box with all torch boxes
    for i, bbox_ref in enumerate(ref_boxes):
        # Handle different data structures
        bbox_ref_array = bbox_ref[0] if isinstance(bbox_ref, tuple) else bbox_ref

        for j, bbox_torch in enumerate(torch_boxes):
            # Handle different data structures
            bbox_torch_array = bbox_torch[0] if isinstance(bbox_torch, tuple) else bbox_torch

            does_pass, pcc_value = check_with_pcc(
                bbox_ref_array, bbox_torch_array, 0.0
            )  # Use 0.0 threshold to get raw PCC
            print(f"PCC value: {pcc_value}")
            print(f"PCC passed: {does_pass}")
            pcc_scores.append((i, j, pcc_value))

    # Sort by PCC descending (best first)
    pcc_scores.sort(key=lambda x: x[2], reverse=True)

    # Take top len(ref_boxes) scores
    top_pcc = pcc_scores[: len(ref_boxes)]

    return top_pcc, pcc_scores


def print_results(top_pcc, all_pcc_scores):
    """
    Print the results in a formatted way.
    """
    print("\n" + "=" * 60)
    print("TOP PCC SCORES (Top len(ref_boxes) matches)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Ref_Idx':<8} {'Torch_Idx':<10} {'PCC_Score':<12}")
    print("-" * 60)

    for rank, (ref_idx, torch_idx, pcc_val) in enumerate(top_pcc, 1):
        # Convert pcc_val to float if it's a string
        try:
            pcc_float = float(pcc_val)
            print(f"{rank:<6} {ref_idx:<8} {torch_idx:<10} {pcc_float:<12.6f}")
        except (ValueError, TypeError):
            print(f"{rank:<6} {ref_idx:<8} {torch_idx:<10} {str(pcc_val):<12}")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total comparisons: {len(all_pcc_scores)}")
    print(f"Top matches shown: {len(top_pcc)}")

    if all_pcc_scores:
        all_pcc_values = [float(score[2]) for score in all_pcc_scores]
        print(f"Best PCC score: {max(all_pcc_values):.6f}")
        print(f"Worst PCC score: {min(all_pcc_values):.6f}")
        print(f"Average PCC score: {np.mean(all_pcc_values):.6f}")
        print(f"Median PCC score: {np.median(all_pcc_values):.6f}")

    print("\n" + "=" * 60)
    print("DETAILED TOP MATCHES")
    print("=" * 60)
    for rank, (ref_idx, torch_idx, pcc_val) in enumerate(top_pcc, 1):
        # Convert pcc_val to float if it's a string
        try:
            pcc_float = float(pcc_val)
            print(f"Rank {rank}: Ref box {ref_idx} ↔ Torch box {torch_idx} (PCC: {pcc_float:.6f})")
        except (ValueError, TypeError):
            print(f"Rank {rank}: Ref box {ref_idx} ↔ Torch box {torch_idx} (PCC: {str(pcc_val)})")


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
    # Load the saved demo inputs
    inputs = torch.load("models/experimental/transfuser/tests/transfuser_inputs_final.pt")

    # Extract each component
    image = inputs["image"]  # RGB camera image tensor
    lidar_bev = inputs["lidar"]  # LiDAR BEV tensor
    velocity = inputs["velocity"]  # Ego velocity tensor
    target_point = inputs["target_point"]  # Target point tensor

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

    ref_feature, pred_wp, ref_head_results, ref_boxes, ref_rotated_bboxes = ref_layer.forward_ego(
        image, lidar_bev, target_point, velocity
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
    ) = ref_head_results

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

    # Convert input to TTNN format
    tt_image_input = ttnn.from_torch(
        image.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_lidar_input = ttnn.from_torch(
        lidar_bev.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_velocity_input = ttnn.from_torch(
        velocity,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_image = ttnn.to_device(tt_image_input, device)
    tt_lidar_bev = ttnn.to_device(tt_lidar_input, device)
    tt_velocity = ttnn.to_device(tt_velocity_input, device)

    tt_features, tt_pred_wp = tt_layer.forward_ego(tt_image, tt_lidar_bev, tt_velocity, target_point)

    torch_feature = ttnn.to_torch(tt_features[0], device=device, dtype=torch.float32)
    # Permute NHWC -> NCHW
    torch_feature = torch_feature.permute(0, 3, 1, 2)

    pcc_passed, pcc_msg = check_with_pcc(ref_feature, torch_feature, pcc=0.95)
    logger.info(f"Feature PCC: {pcc_msg}")
    assert pcc_passed, f"Feature PCC check failed: {pcc_msg}"

    does_pass, pred_wp_pcc_message = check_with_pcc(pred_wp, tt_pred_wp, 0.80)
    logger.info(f"pred wp PCC: {pred_wp_pcc_message}")
    assert does_pass, f"pred wp PCC check failed: {pred_wp_pcc_message}"

    torch_results = ref_layer.head([torch_feature])

    # import pdb; pdb.set_trace()
    does_pass, results_pcc_message = check_with_pcc(ref_head_results[0][0], torch_results[0][0], 0.80)
    logger.info(f"results PCC: {results_pcc_message}")
    assert does_pass, f"results PCC check failed: {results_pcc_message}"

    # Unpack list outputs
    (
        torch_center_heatmap_list,
        torch_wh_list,
        torch_offset_list,
        torch_yaw_class_list,
        torch_yaw_res_list,
        torch_velocity_list,
        torch_brake_list,
    ) = torch_results

    # Extract single tensors from lists
    torch_center_heatmap = torch_center_heatmap_list[0]
    torch_wh = torch_wh_list[0]
    torch_offset = torch_offset_list[0]
    torch_yaw_class = torch_yaw_class_list[0]
    torch_yaw_res = torch_yaw_res_list[0]
    torch_velocity = torch_velocity_list[0]
    torch_brake = torch_brake_list[0]

    # # After the pred_wp PCC check, add bbox post-processing for TTNN outputs

    # Convert TTNN outputs to torch for get_bboxes (it expects torch tensors)
    tt_preds_torch = (
        [torch_center_heatmap],
        [torch_wh],
        [torch_offset],
        [torch_yaw_class],
        [torch_yaw_res],
        [torch_velocity],
        [torch_brake],
    )

    # # Call get_bboxes on the reference head (reusing the same logic)
    torch_boxes = ref_layer.head.get_bboxes(*tt_preds_torch)
    does_pass, box_pcc_message = check_with_pcc(ref_boxes[0][0], torch_boxes[0][0], 0.80)
    logger.info(f"box PCC: {box_pcc_message}")
    torch_bboxes, _ = torch_boxes[0]

    # Filter by confidence threshold
    torch_bboxes = torch_bboxes[torch_bboxes[:, -1] > config.bb_confidence_threshold]

    # Convert to metric coordinates
    torch_rotated_bboxes = []
    for bbox in torch_bboxes.detach().cpu().numpy():
        bbox_metric = ref_layer.get_bbox_local_metric(bbox)
        torch_rotated_bboxes.append(bbox_metric)

    # Compare bbox counts
    logger.info(f"Reference bboxes count: {len(ref_rotated_bboxes)}")
    logger.info(f"TTNN bboxes count: {len(torch_rotated_bboxes)}")

    box_match = len(ref_rotated_bboxes) == len(torch_rotated_bboxes)
    logger.info(f"Box match: {box_match}")

    top_pcc, all_pcc_scores = compare_boxes_pcc(ref_rotated_bboxes, torch_rotated_bboxes)

    print_results(top_pcc, all_pcc_scores)

    does_pass, wh_pcc_message = check_with_pcc(ref_wh, torch_wh, 0.80)
    logger.info(f"WH PCC: {wh_pcc_message}")

    does_pass, offset_pcc_message = check_with_pcc(ref_offset, torch_offset, 0.80)
    logger.info(f"Offset PCC: {offset_pcc_message}")

    does_pass, yaw_class_pcc_message = check_with_pcc(ref_yaw_class, torch_yaw_class, 0.80)
    logger.info(f"Yaw Class PCC: {yaw_class_pcc_message}")

    does_pass, yaw_res_pcc_message = check_with_pcc(ref_yaw_res, torch_yaw_res, 0.80)
    logger.info(f"Yaw Residual PCC: {yaw_res_pcc_message}")

    does_pass, velocity_pcc_message = check_with_pcc(ref_velocity, torch_velocity, 0.80)
    logger.info(f"Velocity PCC: {velocity_pcc_message}")

    does_pass, brake_pcc_message = check_with_pcc(ref_brake, torch_brake, 0.80)
    logger.info(f"Brake PCC: {brake_pcc_message}")

    does_pass, heatmap_pcc_message = check_with_pcc(ref_center_heatmap, torch_center_heatmap, 0.80)
    logger.info(f"Center Heatmap PCC: {heatmap_pcc_message}")

    assert does_pass, f"Center Heatmap PCC Failed! PCC: {heatmap_pcc_message}"

    if does_pass:
        try:
            print("SEED: ", torch.seed())
        except:
            pass
        logger.info("LidarCenterNet Passed!")
    else:
        logger.warning("LidarCenterNet Failed!")
