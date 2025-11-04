# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import os
import torch
import numpy as np
import ttnn
from collections import OrderedDict
from typing import Dict, Any, List, Tuple

from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet, process_input
from models.experimental.transfuser.tt.lidar_center_net import LidarCenterNet as TtLidarCenterNet
from models.experimental.transfuser.tests.test_gpt import create_gpt_preprocessor

from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


# ============================================================
# Helpers
# ============================================================


def create_lidar_center_net_head_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        for head_name in [
            "heatmap_head",
            "wh_head",
            "offset_head",
            "yaw_class_head",
            "yaw_res_head",
            "velocity_head",
            "brake_head",
        ]:
            local_dtype = weight_dtype
            if head_name == "heatmap_head":
                local_dtype = ttnn.float32

            if hasattr(torch_model, head_name):
                head = getattr(torch_model, head_name)
                parameters[head_name] = {}
                parameters[head_name]["conv1_weight"] = ttnn.from_torch(
                    head[0].weight, dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv1_bias"] = ttnn.from_torch(
                    head[0].bias.reshape(1, 1, 1, -1), dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_weight"] = ttnn.from_torch(
                    head[2].weight, dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_bias"] = ttnn.from_torch(
                    head[2].bias.reshape(1, 1, 1, -1), dtype=local_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

        return parameters

    return custom_preprocessor


def get_mesh_mappers(device):
    # Single-device path returns no sharding/concat mappers
    try:
        num = device.get_num_devices()
    except Exception:
        # Some builds expose .num_devices
        num = getattr(device, "num_devices", 1)

    if num != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


def compare_boxes_pcc(ref_boxes, torch_boxes) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    pcc_scores = []
    for i, bbox_ref in enumerate(ref_boxes):
        bbox_ref_array = bbox_ref[0] if isinstance(bbox_ref, tuple) else bbox_ref
        for j, bbox_torch in enumerate(torch_boxes):
            bbox_torch_array = bbox_torch[0] if isinstance(bbox_torch, tuple) else bbox_torch
            _, pcc_value = check_with_pcc(bbox_ref_array, bbox_torch_array, 0.0)
            pcc_scores.append((i, j, float(pcc_value)))
    pcc_scores.sort(key=lambda x: x[2], reverse=True)
    top_pcc = pcc_scores[: len(ref_boxes)]
    return top_pcc, pcc_scores


def print_results(top_pcc, all_pcc_scores):
    print("\n" + "=" * 60)
    print("TOP PCC SCORES (Top len(ref_boxes) matches)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Ref_Idx':<8} {'Torch_Idx':<10} {'PCC_Score':<12}")
    print("-" * 60)
    for rank, (ref_idx, torch_idx, pcc_val) in enumerate(top_pcc, 1):
        print(f"{rank:<6} {ref_idx:<8} {torch_idx:<10} {float(pcc_val):<12.6f}")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total comparisons: {len(all_pcc_scores)}")
    print(f"Top matches shown: {len(top_pcc)}")
    if all_pcc_scores:
        all_vals = [float(s[2]) for s in all_pcc_scores]
        print(f"Best PCC score: {max(all_vals):.6f}")
        print(f"Worst PCC score: {min(all_vals):.6f}")
        print(f"Average PCC score: {np.mean(all_vals):.6f}")
        print(f"Median PCC score: {np.median(all_vals):.6f}")

    print("\n" + "=" * 60)
    print("DETAILED TOP MATCHES")
    print("=" * 60)
    for rank, (ref_idx, torch_idx, pcc_val) in enumerate(top_pcc, 1):
        print(f"Rank {rank}: Ref box {ref_idx} ↔ Torch box {torch_idx} (PCC: {float(pcc_val):.6f})")


def fix_and_filter_checkpoint_keys(
    checkpoint_path: str, target_prefix: str = "module._model.", state_dict_key: str = None
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state_dict = (
        checkpoint[state_dict_key] if state_dict_key and state_dict_key in checkpoint else checkpoint
    )
    new_state_dict = OrderedDict()
    for k, v in checkpoint_state_dict.items():
        if k.startswith(target_prefix):
            name = k[len(target_prefix) :]
            new_state_dict[name] = v
    return new_state_dict


def load_trained_weights(weight_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module._model."):
            clean_key = key[len("module._model.") :]
            state_dict[clean_key] = value
        else:
            state_dict[key] = value

    backbone_keys = [
        "image_encoder",
        "lidar_encoder",
        "transformer1",
        "transformer2",
        "transformer3",
        "transformer4",
        "change_channel_conv_image",
        "change_channel_conv_lidar",
        "up_conv5",
        "up_conv4",
        "up_conv3",
        "c5_conv",
    ]
    for key in list(state_dict.keys()):
        for backbone in backbone_keys:
            if key.startswith(f"{backbone}."):
                new_key = f"_model.{backbone}.{key[len(backbone)+1:]}"
                state_dict[new_key] = state_dict.pop(key)
                break

    detection_components = ["head", "pred_bev", "join", "decoder", "output"]
    for key in list(state_dict.keys()):
        for component in detection_components:
            if key.startswith(f"module.{component}."):
                new_key = key[len("module.") :]
                state_dict[new_key] = state_dict.pop(key)
                break

    return state_dict


def delete_incompatible_keys(state_dict: Dict[str, Any], keys_to_delete: List[str]) -> Dict[str, Any]:
    new_state = OrderedDict(state_dict)
    for k in keys_to_delete:
        if k in new_state:
            del new_state[k]
    return new_state


def open_tt_device(device_id: int = 0, l1_small_size: int = 16384, trace_region_size: int = 0, worker_l1_size: int = 0):
    """
    Open a single TT device using the device-id API exposed by this TTNN build.
    Returns a MeshDevice-like handle with one device.
    """
    return ttnn.open_device(
        device_id=device_id,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
        worker_l1_size=worker_l1_size,
    )


# ============================================================
# Demo main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="LidarCenterNet TTNN vs Torch demo (no pytest).")
    parser.add_argument("--data-root", type=str, required=True, help="Folder with scenario data (images/lidar).")
    parser.add_argument("--frame", type=str, required=True, help="Frame id inside data_root, e.g., 0120")
    parser.add_argument("--weights", type=str, required=True, help="Path to Transfuser weight file (.pth)")
    parser.add_argument("--image-arch", type=str, default="regnety_032")
    parser.add_argument("--lidar-arch", type=str, default="regnety_032")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # Fallback flags
    parser.add_argument("--no-fallback", action="store_true", help="Disable TTNN fallback paths.")
    parser.add_argument(
        "--use-optimized-self-attn",
        action="store_true",
        default=False,
        help="Enable optimized self-attention in GPT preprocessors.",
    )

    # Device options (single device API)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--l1-small-size", type=int, default=16384)
    parser.add_argument("--trace-region-size", type=int, default=0)
    parser.add_argument("--worker-l1-size", type=int, default=0)

    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # Basic checks
    if not os.path.isdir(args.data_root):
        logger.error(f"data_root not found: {args.data_root}")
        sys.exit(1)
    if not os.path.isfile(args.weights):
        logger.error(f"weights file not found: {args.weights}")
        sys.exit(1)

    device = None
    try:
        # Config + inputs
        config = GlobalConfig(setting="eval")
        config.n_layer = args.layers
        config.use_target_point_image = True

        logger.info(f"Loading inputs from {args.data_root}, frame {args.frame}")
        inputs = process_input(args.data_root, args.frame, config=config, normalize_image=False)

        image = inputs["image"]
        lidar_bev = inputs["lidar"]
        velocity = inputs["velocity"]
        target_point = inputs["target_point"]

        # Device
        logger.info("Opening TTNN device (single device API)...")
        device = open_tt_device(
            device_id=args.device_id,
            l1_small_size=args.l1_small_size,
            trace_region_size=args.trace_region_size,
            worker_l1_size=args.worker_l1_size,
        )
        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

        # Build Torch reference model and load weights
        logger.info("Building Torch reference model...")
        ref_layer = LidarCenterNet(
            config,
            backbone="transFuser",
            image_architecture=args.image_arch,
            lidar_architecture=args.lidar_arch,
            use_velocity=False,
        ).eval()

        logger.info(f"Loading and cleaning weights from: {args.weights}")
        modified_state_dict = load_trained_weights(args.weights)
        modified_state_dict = delete_incompatible_keys(
            modified_state_dict,
            [
                "_model.lidar_encoder._model.stem.conv.weight",
                "module.seg_decoder.deconv1.0.weight",
                "module.seg_decoder.deconv1.0.bias",
                "module.seg_decoder.deconv1.2.weight",
                "module.seg_decoder.deconv1.2.bias",
                "module.seg_decoder.deconv2.0.weight",
                "module.seg_decoder.deconv2.0.bias",
                "module.seg_decoder.deconv2.2.weight",
                "module.seg_decoder.deconv2.2.bias",
                "module.seg_decoder.deconv3.0.weight",
                "module.seg_decoder.deconv3.0.bias",
                "module.seg_decoder.deconv3.2.weight",
                "module.seg_decoder.deconv3.2.bias",
                "module.depth_decoder.deconv1.0.weight",
                "module.depth_decoder.deconv1.0.bias",
                "module.depth_decoder.deconv1.2.weight",
                "module.depth_decoder.deconv1.2.bias",
                "module.depth_decoder.deconv2.0.weight",
                "module.depth_decoder.deconv2.0.bias",
                "module.depth_decoder.deconv2.2.weight",
                "module.depth_decoder.deconv2.2.bias",
                "module.depth_decoder.deconv3.0.weight",
                "module.depth_decoder.deconv3.0.bias",
                "module.depth_decoder.deconv3.2.weight",
                "module.depth_decoder.deconv3.2.bias",
            ],
        )
        ref_layer.load_state_dict(modified_state_dict, strict=True)

        # Torch forward (reference)
        logger.info("Running Torch reference forward...")
        (
            ref_fused_features,
            ref_feature,
            pred_wp,
            ref_head_results,
            ref_boxes,
            ref_rotated_bboxes,
        ) = ref_layer.forward_ego(image, lidar_bev, target_point, velocity)

        (
            ref_center_heatmap_list,
            ref_wh_list,
            ref_offset_list,
            ref_yaw_class_list,
            ref_yaw_res_list,
            ref_velocity_list,
            ref_brake_list,
        ) = ref_head_results

        ref_center_heatmap = ref_center_heatmap_list[0]
        ref_wh = ref_wh_list[0]
        ref_offset = ref_offset_list[0]
        ref_yaw_class = ref_yaw_class_list[0]
        ref_yaw_res = ref_yaw_res_list[0]
        ref_velocity = ref_velocity_list[0]
        ref_brake = ref_brake_list[0]
        torch_model = ref_layer._model

        # Preprocess parameters for TTNN
        logger.info("Preprocessing parameters for TTNN...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=None,
        )

        # GPT submodules
        for name in ["transformer1", "transformer2", "transformer3", "transformer4"]:
            parameters[name] = preprocess_model_parameters(
                initialize_model=lambda n=name: getattr(torch_model, n),
                custom_preprocessor=create_gpt_preprocessor(
                    device, args.layers, ttnn.bfloat16, args.use_optimized_self_attn
                ),
                device=device,
            )

        # Head
        parameters["head"] = preprocess_model_parameters(
            initialize_model=lambda: ref_layer.head,
            custom_preprocessor=create_lidar_center_net_head_preprocessor(device, ttnn.bfloat16),
            device=device,
        )

        transfuser_model = ref_layer._model
        tt_layer = TtLidarCenterNet(
            device,
            parameters,
            config,
            backbone="transFuser",
            torch_model=transfuser_model,
            use_fallback=(not args.no_fallback),
        )

        # Convert inputs to TTNN
        logger.info("Converting inputs to TTNN tensors...")
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
        tt_velocity_input = ttnn.from_torch(velocity, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        tt_image = ttnn.to_device(tt_image_input, device)
        tt_lidar_bev = ttnn.to_device(tt_lidar_input, device)
        tt_velocity = ttnn.to_device(tt_velocity_input, device)

        # TTNN forward
        logger.info("Running TTNN forward...")
        tt_features, tt_fused_features = tt_layer.forward_ego(tt_image, tt_lidar_bev, tt_velocity, target_point)

        tt_fused_torch = ttnn.to_torch(tt_fused_features, device=device)
        _, fused_features_pcc_message = check_with_pcc(ref_fused_features, tt_fused_torch, 0.97)
        logger.info(f"Fused features PCC: {fused_features_pcc_message}")

        # Waypoints via GRU (Torch path)
        tt_fused_torch_fp32 = tt_fused_torch.to(torch.float32)
        tt_pred_wp, _, _, _, _ = ref_layer.forward_gru(tt_fused_torch_fp32, target_point)
        _, pred_wp_pcc_message = check_with_pcc(pred_wp, tt_pred_wp, 0.97)
        logger.info(f"Pred waypoints PCC: {pred_wp_pcc_message}")

        # Feature compare
        tt_feature_0 = ttnn.to_torch(tt_features[0], device=device).to(torch.float32).permute(0, 3, 1, 2)
        _, pcc_msg = check_with_pcc(ref_feature, tt_feature_0, 0.97)
        logger.info(f"Feature PCC: {pcc_msg}")

        # Run head on Torch with TT feature
        torch_results = ref_layer.head([tt_feature_0])
        _, results_pcc_message = check_with_pcc(ref_head_results[0][0], torch_results[0][0], 0.97)
        logger.info(f"Head (results) PCC: {results_pcc_message}")

        (
            torch_center_heatmap_list,
            torch_wh_list,
            torch_offset_list,
            torch_yaw_class_list,
            torch_yaw_res_list,
            torch_velocity_list,
            torch_brake_list,
        ) = torch_results

        torch_center_heatmap = torch_center_heatmap_list[0]
        torch_wh = torch_wh_list[0]
        torch_offset = torch_offset_list[0]
        torch_yaw_class = torch_yaw_class_list[0]
        torch_yaw_res = torch_yaw_res_list[0]
        torch_velocity = torch_velocity_list[0]
        torch_brake = torch_brake_list[0]

        # BBoxes
        torch_boxes = ref_layer.head.get_bboxes(
            [torch_center_heatmap],
            [torch_wh],
            [torch_offset],
            [torch_yaw_class],
            [torch_yaw_res],
            [torch_velocity],
            [torch_brake],
        )
        _, box_pcc_message = check_with_pcc(ref_boxes[0][0], torch_boxes[0][0], 0.80)
        logger.info(f"Box PCC: {box_pcc_message}")

        torch_bboxes, _ = torch_boxes[0]
        torch_bboxes = torch_bboxes[torch_bboxes[:, -1] > config.bb_confidence_threshold]

        torch_rotated_bboxes = []
        for bbox in torch_bboxes.detach().cpu().numpy():
            torch_rotated_bboxes.append(ref_layer.get_bbox_local_metric(bbox))

        logger.info(f"Reference bboxes count: {len(ref_rotated_bboxes)}")
        logger.info(f"TTNN bboxes count: {len(torch_rotated_bboxes)}")
        logger.info(f"Box count match: {len(ref_rotated_bboxes) == len(torch_rotated_bboxes)}")

        top_pcc, all_pcc_scores = compare_boxes_pcc(ref_rotated_bboxes, torch_rotated_bboxes)
        print_results(top_pcc, all_pcc_scores)

        # Per-head PCCs
        _, wh_pcc_message = check_with_pcc(ref_wh, torch_wh, 0.97)
        _, offset_pcc_message = check_with_pcc(ref_offset, torch_offset, 0.97)
        _, yaw_class_pcc_message = check_with_pcc(ref_yaw_class, torch_yaw_class, 0.97)
        _, yaw_res_pcc_message = check_with_pcc(ref_yaw_res, torch_yaw_res, 0.97)
        _, velocity_pcc_message = check_with_pcc(ref_velocity, torch_velocity, 0.97)
        _, brake_pcc_message = check_with_pcc(ref_brake, torch_brake, 0.97)
        _, heatmap_pcc_message = check_with_pcc(ref_center_heatmap, torch_center_heatmap, 0.97)

        print("\n=== Per-head PCC Summary ===")
        print(f"WH:        {wh_pcc_message}")
        print(f"Offset:    {offset_pcc_message}")
        print(f"Yaw Class: {yaw_class_pcc_message}")
        print(f"Yaw Res:   {yaw_res_pcc_message}")
        print(f"Velocity:  {velocity_pcc_message}")
        print(f"Brake:     {brake_pcc_message}")
        print(f"Heatmap:   {heatmap_pcc_message}")

        print("\nDemo complete.")

    finally:
        # Best-effort close
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass


if __name__ == "__main__":
    main()
