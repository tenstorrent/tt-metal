# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
import cv2
from models.experimental.BEVFormerV2.tt.ttnn_bevformer_v2 import TtBevFormerV2
from models.experimental.BEVFormerV2.reference.bevformer_v2 import BEVFormerV2

import ttnn
import gc
from models.experimental.BEVFormerV2.common import load_torch_model
from models.experimental.BEVFormerV2.tt.model_preprocessing import create_bevformerv2_model_parameters
from models.experimental.BEVFormerV2.demo.demo_data_loader import load_demo_data


Quaternion = None
try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None
    print(
        "Warning: pyquaternion not found in virtual environment. "
        "Coordinate transformations will use lidar space (not converted to global). "
        "Install with: pip install pyquaternion"
    )

# Class order must match the reference config: bevformerv2-r50-t1-base-24ep.py
CLASSES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test BEVFormerV2 model")
    parser.add_argument(
        "--checkpoint", default=None, help="checkpoint file (deprecated: using load_torch_model instead)"
    )
    parser.add_argument(
        "--data-root", default="models/experimental/BEVFormerV2/demo/demo_data", help="data root directory"
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="sample index to test (default: 0, use -1 for all)")
    parser.add_argument(
        "--out", default="models/experimental/BEVFormerV2/demo/outputs/results.json", help="output result file"
    )
    parser.add_argument("--eval", action="store_true", help="run evaluation")
    return parser.parse_args()


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img.astype(np.float32)


def normalize_image(img, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False):
    img = img.copy()
    if to_rgb:
        img = img[..., ::-1]
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    return img


def pad_image(img, size_divisor=32):
    h, w = img.shape[:2]
    pad_h = (size_divisor - h % size_divisor) % size_divisor
    pad_w = (size_divisor - w % size_divisor) % size_divisor
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    return img


def crop_resize_flip_image(img, crop=(0, 260, 1600, 900), resize_h=640, flip=False):
    x1, y1, x2, y2 = crop
    img_pil = Image.fromarray(np.uint8(img))
    img_pil = img_pil.crop(crop)

    original_h = y2 - y1
    resize = resize_h / original_h
    resize_w = int((x2 - x1) * resize)
    resize_dims = (resize_w, resize_h)

    img_pil = img_pil.resize(resize_dims, Image.BILINEAR)
    if flip:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    ida_rot = np.eye(2) * resize
    ida_tran = -np.array(crop[:2]) * resize
    ida_mat = np.eye(3)
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 2] = ida_tran

    img = np.array(img_pil).astype(np.float32)
    return img, ida_mat, resize_dims


def prepare_sample_data(info, data_root):
    cam_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    img_paths = []
    lidar2cam_rts = []
    cam2img_list = []
    ori_img_shapes = []

    for cam_type in cam_types:
        if cam_type not in info["cams"]:
            continue
        cam_info = info["cams"][cam_type]
        data_path = cam_info["data_path"]

        if data_path.startswith("./data/nuscenes"):
            remaining_path = data_path[len("./data/nuscenes") :].lstrip("/")
            full_path = os.path.join(data_root, remaining_path)
        elif data_path.startswith("./data/"):
            remaining_path = data_path[len("./data/") :].lstrip("/")
            full_path = os.path.join(data_root, remaining_path)
        elif not os.path.isabs(data_path):
            full_path = os.path.join(data_root, data_path)
        else:
            full_path = data_path

        img = load_image(full_path)
        ori_img_shapes.append(img.shape[:2])

        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = cam_info["cam_intrinsic"]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

        img_paths.append(full_path)
        cam2img_list.append(viewpad.copy())
        lidar2cam_rts.append(lidar2cam_rt.T)

    ida_aug_conf_eval = {
        "reisze": [256],
        "crop": (0, 260, 1600, 900),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
    crop = ida_aug_conf_eval["crop"]
    resized_h = 256
    resize_w = 704
    resize_dims = (resize_w, resized_h)
    flip = False

    processed_imgs = []
    img_shapes = []
    lidar2img_rts = []

    for i, (img_path, cam2img, lidar2cam) in enumerate(zip(img_paths, cam2img_list, lidar2cam_rts)):
        img = load_image(img_path)
        img, ida_mat, actual_resize_dims = crop_resize_flip_image(img, crop=crop, resize_h=resized_h, flip=flip)

        cam2img[:3, :3] = np.matmul(ida_mat, cam2img[:3, :3])
        lidar2img_rt = np.matmul(cam2img, lidar2cam)

        img = normalize_image(img, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
        img = pad_image(img, size_divisor=32)

        img_shapes.append(img.shape[:2])
        lidar2img_rts.append(lidar2img_rt)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if len(processed_imgs) == 0:
            processed_imgs = img.unsqueeze(0)
        else:
            processed_imgs = torch.cat([processed_imgs, img.unsqueeze(0)], dim=0)

    if processed_imgs.dim() == 4:
        processed_imgs = processed_imgs.unsqueeze(0)

    can_bus = info.get("can_bus", np.zeros(18))
    ego2global_translation = info.get("ego2global_translation", np.zeros(3))
    ego2global_rotation = info.get("ego2global_rotation", np.eye(3))
    lidar2ego_translation = info.get("lidar2ego_translation", np.zeros(3))
    lidar2ego_rotation = info.get("lidar2ego_rotation", np.eye(3))
    timestamp = info.get("timestamp", 0.0) / 1e6

    img_metas = [
        {
            "sample_idx": info.get("token", "unknown"),
            "img_shape": img_shapes,
            "ori_shape": ori_img_shapes,
            "lidar2img": [lidar2img_rt.tolist() for lidar2img_rt in lidar2img_rts],
            "can_bus": can_bus.tolist() if isinstance(can_bus, np.ndarray) else can_bus,
            "ego2global_translation": ego2global_translation.tolist()
            if isinstance(ego2global_translation, np.ndarray)
            else ego2global_translation,
            "ego2global_rotation": ego2global_rotation.tolist()
            if isinstance(ego2global_rotation, np.ndarray)
            else ego2global_rotation,
            "lidar2ego_translation": lidar2ego_translation.tolist()
            if isinstance(lidar2ego_translation, np.ndarray)
            else lidar2ego_translation,
            "lidar2ego_rotation": lidar2ego_rotation.tolist()
            if isinstance(lidar2ego_rotation, np.ndarray)
            else lidar2ego_rotation,
            "timestamp": timestamp,
        }
    ]

    return processed_imgs, img_metas


def format_results_to_json(results, infos, output_path):
    nusc_annos = {}

    for sample_id, result in enumerate(results):
        if sample_id >= len(infos):
            continue

        info = infos[sample_id]
        sample_token = info.get("token", f"sample_{sample_id}")

        if "pts_bbox" not in result:
            continue

        pts_bbox = result["pts_bbox"]
        boxes_3d = pts_bbox.get("boxes_3d", None)
        scores_3d = pts_bbox.get("scores_3d", None)
        labels_3d = pts_bbox.get("labels_3d", None)

        if boxes_3d is None or scores_3d is None or labels_3d is None:
            nusc_annos[sample_token] = []
            continue

        if isinstance(boxes_3d, torch.Tensor):
            boxes_3d = boxes_3d.cpu().numpy()
        if isinstance(scores_3d, torch.Tensor):
            scores_3d = scores_3d.cpu().numpy()
        if isinstance(labels_3d, torch.Tensor):
            labels_3d = labels_3d.cpu().numpy()

        annos = []

        for i in range(len(boxes_3d)):
            box = boxes_3d[i]
            score = float(scores_3d[i])
            label = int(labels_3d[i])

            if label < 0 or label >= len(CLASSES):
                continue

            cx, cy, cz = box[0], box[1], box[2]
            w, l, h = box[3], box[4], box[5]
            rot = box[6]

            if len(box) > 8:
                vx, vy = box[7], box[8]
            else:
                vx, vy = 0.0, 0.0

            # The box coordinates from bbox_coder are in bottom-center format
            # The reference applies z adjustment in get_bboxes, but then gravity_center adds h/2 back
            # So the net effect is using bottom-center coordinates directly
            # This matches the reference's behavior where box3d.gravity_center computes center from bottom_center
            center_lidar = np.array([cx, cy, cz])

            try:
                if Quaternion is not None:
                    # Handle quaternion format (4 elements) vs rotation matrix (3x3)
                    lidar2ego_rot_data = np.array(info["lidar2ego_rotation"])
                    ego2global_rot_data = np.array(info["ego2global_rotation"])

                    # Check if it's a quaternion (4 elements) or rotation matrix (3x3)
                    if lidar2ego_rot_data.shape == (4,):
                        lidar2ego_rot = Quaternion(lidar2ego_rot_data).rotation_matrix
                    elif lidar2ego_rot_data.shape == (3, 3):
                        lidar2ego_rot = lidar2ego_rot_data
                    else:
                        raise ValueError(f"Unexpected lidar2ego_rotation shape: {lidar2ego_rot_data.shape}")

                    if ego2global_rot_data.shape == (4,):
                        ego2global_rot = Quaternion(ego2global_rot_data).rotation_matrix
                    elif ego2global_rot_data.shape == (3, 3):
                        ego2global_rot = ego2global_rot_data
                    else:
                        raise ValueError(f"Unexpected ego2global_rotation shape: {ego2global_rot_data.shape}")

                    lidar2ego_trans = np.array(info["lidar2ego_translation"])
                    ego2global_trans = np.array(info["ego2global_translation"])

                    center_ego = center_lidar @ lidar2ego_rot.T + lidar2ego_trans
                    center_global = center_ego @ ego2global_rot.T + ego2global_trans

                    vel_lidar = np.array([vx, 0.0, vy])
                    vel_ego = vel_lidar @ lidar2ego_rot.T
                    vel_global = vel_ego @ ego2global_rot.T
                    velocity = [float(vel_global[0]), float(vel_global[1])]
                else:
                    if sample_id == 0 and i == 0:  # Print warning only once per sample
                        print(
                            f"Warning: pyquaternion not available. Coordinates for sample {sample_token} "
                            "are in lidar space (not converted to global). "
                            "Install with: pip install pyquaternion"
                        )
                    center_global = center_lidar
                    velocity = [float(vx), float(vy)]
            except Exception as e:
                if sample_id == 0 and i == 0:  # Print warning only once per sample
                    print(
                        f"Warning: Failed to convert coordinates for sample {sample_token}: {e}. "
                        "Using lidar coordinates (not converted to global)."
                    )
                center_global = center_lidar
                velocity = [float(vx), float(vy)]

            dims = np.array([w, l, h])
            dims[[0, 1, 2]] = dims[[2, 0, 1]]

            yaw = -rot

            try:
                if Quaternion is not None:
                    q1 = Quaternion(axis=[0, 0, 1], radians=yaw)
                    q2 = Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
                    quat = q2 * q1
                    rotation = quat.elements.tolist()
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            except Exception:
                rotation = [0.0, 0.0, 0.0, 1.0]

            detection_name = CLASSES[label]

            anno = {
                "sample_token": sample_token,
                "translation": center_global.tolist(),
                "size": dims.tolist(),
                "rotation": rotation,
                "velocity": velocity,
                "detection_name": detection_name,
                "detection_score": score,
                "attribute_name": "",
            }
            annos.append(anno)

        nusc_annos[sample_token] = annos

    nusc_submissions = {
        "meta": {"use_lidar": False, "use_camera": True, "use_radar": False, "use_map": False, "use_external": False},
        "results": nusc_annos,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(nusc_submissions, f, indent=2)

    return output_path


def main():
    args = parse_args()

    device = ttnn.open_device(device_id=0, l1_small_size=4 * 8192)

    torch_model = BEVFormerV2(
        use_grid_mask=False,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(bev_h=100, bev_w=100, num_query=900, num_classes=10, in_channels=256),
        video_test_mode=True,
    )

    print("Loading weights using load_torch_model...")
    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=None)

    for m in torch_model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    torch_model.pts_bbox_head.transformer.encoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.encoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.encoder.num_layers = 6
    torch_model.pts_bbox_head.transformer.decoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.decoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.decoder.num_layers = 6

    print("Loading demo data (sample 0)")
    infos = load_demo_data(sample_idx=args.sample_idx if args.sample_idx >= 0 else 0)
    print(f"Loaded {len(infos)} samples")

    if args.sample_idx >= len(infos):
        print(f"Error: sample_idx {args.sample_idx} >= {len(infos)}")
        return

    if args.sample_idx < 0:
        sample_indices = range(len(infos))
    else:
        sample_indices = [args.sample_idx]

    first_sample = True
    outputs = []

    for idx in sample_indices:
        info = infos[idx]
        print(f"Processing sample {idx}: {info.get('token', 'unknown')}")

        imgs, img_metas = prepare_sample_data(info, args.data_root)

        if first_sample:
            print("Preprocessing parameters for TTNN...")
            if isinstance(imgs, torch.Tensor) and imgs.dim() == 5:
                B, N, C, H, W = imgs.shape
                if B == 1:
                    imgs_for_preprocessing = imgs.squeeze(0)
                else:
                    imgs_for_preprocessing = imgs.reshape(B * N, C, H, W)
            else:
                imgs_for_preprocessing = imgs
            img_list = [imgs_for_preprocessing]
            encoder_num_layers = torch_model.pts_bbox_head.transformer.encoder.num_layers
            decoder_num_layers = torch_model.pts_bbox_head.transformer.decoder.num_layers
            parameters = create_bevformerv2_model_parameters(
                torch_model,
                [
                    False,
                    img_list,
                    img_metas,
                ],
                device,
            )

            del torch_model
            gc.collect()

            print("Creating TTNN model...")
            ttnn_model = TtBevFormerV2(
                device=device,
                params=parameters,
                use_grid_mask=False,
                img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
                img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
                pts_bbox_head=dict(
                    bev_h=100,
                    bev_w=100,
                    num_query=900,
                    num_classes=10,
                    in_channels=256,
                    encoder_num_layers=encoder_num_layers,
                    decoder_num_layers=decoder_num_layers,
                ),
                video_test_mode=True,
            )
            first_sample = False

        print("Converting images to TTNN format...")
        if isinstance(imgs, torch.Tensor) and imgs.dim() == 5:
            B, N, C, H, W = imgs.shape
            if B == 1:
                imgs_torch = imgs.squeeze(0)
            else:
                imgs_torch = imgs.reshape(B * N, C, H, W)
        else:
            imgs_torch = imgs
        imgs_ttnn = ttnn.from_torch(imgs_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        imgs_ttnn = [imgs_ttnn]

        print("Running TTNN inference...")
        with torch.no_grad():
            result = ttnn_model(
                return_loss=False,
                img=imgs_ttnn,
                img_metas=img_metas,
            )[0]

        if isinstance(result, dict):
            outputs.append(result)
        elif isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)

    print(f"\nInference completed. Processed {len(outputs)} samples")
    if len(outputs) > 0 and isinstance(outputs[0], dict):
        print(f"Result keys: {list(outputs[0].keys())}")
        if "pts_bbox" in outputs[0]:
            pts_bbox = outputs[0]["pts_bbox"]
            if "boxes_3d" in pts_bbox:
                num_dets = len(pts_bbox["boxes_3d"]) if hasattr(pts_bbox["boxes_3d"], "__len__") else "N/A"
                print(f"Number of detections: {num_dets}")
            if "scores_3d" in pts_bbox:
                scores = pts_bbox["scores_3d"]
                if hasattr(scores, "min") and hasattr(scores, "max"):
                    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"Score mean: {scores.mean():.4f}, Score std: {scores.std():.4f}")
                    if len(scores) > 0:
                        top_scores = (
                            torch.topk(scores, min(10, len(scores)))[0]
                            if isinstance(scores, torch.Tensor)
                            else sorted(scores, reverse=True)[:10]
                        )
                        print(f"Top 10 scores: {top_scores}")

    if args.out:
        if args.out.endswith(".json"):
            os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
            json_path = format_results_to_json(outputs, infos, args.out)
            print(f"Results saved to JSON: {json_path}")
        else:
            # For non-JSON outputs, save as JSON instead of pickle
            os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
            json_path = args.out.replace(".pkl", ".json") if args.out.endswith(".pkl") else args.out + ".json"
            json_path = format_results_to_json(outputs, infos, json_path)
            print(f"Results saved to JSON: {json_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
