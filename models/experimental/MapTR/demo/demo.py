# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import torch
import ttnn
import json
from models.experimental.MapTR.reference.dependency import Config
from models.experimental.MapTR.reference.dependency import load_checkpoint, wrap_fp16_model
from models.experimental.MapTR.reference.dependency import get_logger, ProgressBar
from models.experimental.MapTR.reference.dependency import build_dataset, build_model

# Import build_dataloader after dependency to avoid circular import
from models.experimental.MapTR.reference.dependency import replace_ImageToTensor
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_maptr import TtMapTR
from models.experimental.MapTR.tt.model_preprocessing import (
    create_maptr_model_parameters,
)
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# Import processing functions from local processing.py
# Add current directory to path to ensure import works when running as script
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from processing import write_map_annotations_to_file, write_sample_info_to_file

CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]
CANDIDATE = [
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
]


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def parse_args():
    parser = argparse.ArgumentParser(description="MapTR TTNN Demo - visualize predictions using TTNN model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=MAPTR_WEIGHTS_PATH,
        help=f"checkpoint file (default: {MAPTR_WEIGHTS_PATH}, will auto-download if missing)",
    )
    parser.add_argument("--score-thresh", default=0.4, type=float, help="score threshold for predictions")
    parser.add_argument("--show-dir", help="directory where visualizations will be saved")
    parser.add_argument("--show-cam", action="store_true", help="show camera pic")
    parser.add_argument(
        "--gt-format",
        type=str,
        nargs="+",
        default=[
            "fixed_num_pts",
        ],
        help='vis format, default should be "points",' 'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]',
    )
    parser.add_argument(
        "--device-params",
        default='{"l1_small_size": 24576}',
        help="Device parameters as JSON string",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Import dataset classes to register them
    from models.experimental.MapTR.reference import datasets_nuscenes_map  # noqa: F401
    from models.experimental.MapTR.reference import datasets_nuscenes  # noqa: F401

    # Import pipeline classes to register them
    from models.experimental.MapTR.reference import pipelines  # noqa: F401

    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            try:
                if hasattr(cfg, "plugin_dir"):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = plugin_dir.rstrip("/").replace("/", ".")
                    if not _module_dir.startswith("models.experimental.MapTR."):
                        _module_path = "models.experimental.MapTR." + _module_dir
                    else:
                        _module_path = _module_dir
                else:
                    _module_path = "models.experimental.MapTR.projects.mmdet3d_plugin"
                print(f"Importing plugin from: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            except Exception as e:
                print(f"Warning: Failed to import plugin module {_module_path}: {e}")
                print("Trying default plugin path...")
                _module_path = "models.experimental.MapTR.projects.mmdet3d_plugin"
                plg_lib = importlib.import_module(_module_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0], "vis_pred")
    os.makedirs(osp.abspath(args.show_dir), exist_ok=True)

    with open(osp.join(args.show_dir, osp.basename(args.config)), "w") as f:
        json.dump(dict(cfg), f, indent=2)
    mmlogger = get_logger("demo")
    mmlogger.info(f"DONE create vis_pred dir: {args.show_dir}")

    # Generate data files from processing.py instead of loading from external JSON/pickle files
    if write_map_annotations_to_file is not None and write_sample_info_to_file is not None:
        # Handle both dict and object-style config access
        if isinstance(cfg.data.test, dict):
            data_root = cfg.data.test.get("data_root", "models/experimental/MapTR/data/nuscenes/")
            map_ann_file = cfg.data.test.get("map_ann_file")
            ann_file = cfg.data.test.get("ann_file")
        else:
            data_root = getattr(cfg.data.test, "data_root", "models/experimental/MapTR/data/nuscenes/")
            map_ann_file = getattr(cfg.data.test, "map_ann_file", None)
            ann_file = getattr(cfg.data.test, "ann_file", None)

        # Generate and write map annotations JSON file
        if map_ann_file:
            map_ann_file_path = map_ann_file
            # Check if path already includes data_root (from config like data_root + "file.json")
            if not osp.isabs(map_ann_file_path) and not map_ann_file_path.startswith(data_root):
                map_ann_file_path = osp.join(data_root, map_ann_file_path)
            mmlogger.info(f"Generating map annotations from processing.py and writing to {map_ann_file_path}")
            write_map_annotations_to_file(map_ann_file_path)

        # Generate and write sample info pickle file
        if ann_file:
            ann_file_path = ann_file
            # Check if path already includes data_root (from config like data_root + "file.pkl")
            if not osp.isabs(ann_file_path) and not ann_file_path.startswith(data_root):
                ann_file_path = osp.join(data_root, ann_file_path)
            mmlogger.info(f"Generating sample info from processing.py and writing to {ann_file_path}")
            write_sample_info_to_file(ann_file_path)
    else:
        mmlogger.warning("Processing functions not available, will use default file loading")

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True  # TODO, this is a hack
    # Import build_dataloader here to avoid circular import
    from models.experimental.MapTR.reference.datasets import build_dataloader

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    mmlogger.info("Done build test data set")

    cfg.model.train_cfg = None
    torch_model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(torch_model)

    # Ensure checkpoint is downloaded if using default path
    checkpoint_path = args.checkpoint
    if checkpoint_path == MAPTR_WEIGHTS_PATH or not os.path.exists(checkpoint_path):
        ensure_checkpoint_downloaded(checkpoint_path)

    mmlogger.info("loading checkpoint")
    checkpoint = load_checkpoint(torch_model, checkpoint_path, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        torch_model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        torch_model.CLASSES = dataset.CLASSES
    if "PALETTE" in checkpoint.get("meta", {}):
        torch_model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        torch_model.PALETTE = dataset.PALETTE
    mmlogger.info("DONE load checkpoint")
    torch_model.eval()

    device_params = json.loads(args.device_params)
    device = ttnn.open_device(device_id=0, l1_small_size=device_params.get("l1_small_size", 24576))
    mmlogger.info("DONE open TTNN device")

    bev_h = cfg.bev_h_ if hasattr(cfg, "bev_h_") else 200
    bev_w = cfg.bev_w_ if hasattr(cfg, "bev_w_") else 100
    pc_range = cfg.point_cloud_range if hasattr(cfg, "point_cloud_range") else [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    num_vec = cfg.model.pts_bbox_head.num_vec if hasattr(cfg.model.pts_bbox_head, "num_vec") else 50
    num_pts_per_vec = (
        cfg.model.pts_bbox_head.num_pts_per_vec if hasattr(cfg.model.pts_bbox_head, "num_pts_per_vec") else 20
    )
    num_classes = cfg.model.pts_bbox_head.num_classes if hasattr(cfg.model.pts_bbox_head, "num_classes") else 3
    embed_dims = cfg.model.pts_bbox_head.in_channels if hasattr(cfg.model.pts_bbox_head, "in_channels") else 256

    expected_img_h, expected_img_w = 384, 640
    if hasattr(cfg, "data") and hasattr(cfg.data, "test") and hasattr(cfg.data.test, "pipeline"):
        for transform in cfg.data.test.pipeline:
            if isinstance(transform, dict) and transform.get("type") == "ResizeMultiview3D":
                img_scale = transform.get("img_scale", (expected_img_h, expected_img_w))
                if isinstance(img_scale, (list, tuple)):
                    expected_img_h, expected_img_w = img_scale[0], img_scale[1]
                break
            elif isinstance(transform, dict) and transform.get("type") == "Resize":
                img_scale = transform.get("img_scale", (expected_img_h, expected_img_w))
                if isinstance(img_scale, (list, tuple)):
                    expected_img_h, expected_img_w = img_scale[0], img_scale[1]
                break
    mmlogger.info(f"Expected input image size: {expected_img_h}x{expected_img_w}")

    mmlogger.info("Creating TTNN model parameters...")
    dummy_input = torch.randn(1, 6, 3, expected_img_h, expected_img_w)
    parameters = create_maptr_model_parameters(
        torch_model,
        dummy_input,
        device,
    )

    mmlogger.info("Creating TTNN MapTR model...")
    tt_model = TtMapTR(
        device=device,
        params=parameters,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_classes=num_classes,
        embed_dims=embed_dims,
    )
    mmlogger.info("DONE create TTNN MapTR model")

    img_norm_cfg = cfg.img_norm_cfg

    mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(img_norm_cfg["std"], dtype=np.float32)
    to_bgr = img_norm_cfg["to_rgb"]

    car_img = Image.open("models/experimental/MapTR/figs/lidar_car.png")
    colors_plt = ["orange", "b", "g"]

    mmlogger.info("BEGIN vis test dataset samples gt label & pred using TTNN model")

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False

    if len(CANDIDATE) == 0:
        for i in range(min(len(dataset), 10)):
            try:
                data_info = dataset.get_data_info(i)
                if "lidar_path" in data_info:
                    lidar_filename = osp.basename(data_info["lidar_path"])
                    sample_name = lidar_filename.replace("__LIDAR_TOP__", "_").split(".")[0]
                    CANDIDATE.append(sample_name)
            except:
                continue

    prog_bar = ProgressBar(len(dataset))

    try:
        for i, data in enumerate(data_loader):
            has_gt = False
            if (
                "gt_labels_3d" in data
                and hasattr(data["gt_labels_3d"], "data")
                and len(data["gt_labels_3d"].data[0]) > 0
            ):
                has_gt = (data["gt_labels_3d"].data[0][0] != -1).any()
            if not has_gt:
                mmlogger.warning(f"\n empty gt for index {i}, will visualize predictions only")

            if "img" not in data:
                mmlogger.warning(
                    f"\n no img in data for index {i}, available keys: {list(data.keys())}, will continue with map visualization only"
                )
                img = None
            else:
                img = data["img"][0].data[0] if len(data["img"][0].data) > 0 else None

            img_metas_extracted = None
            if "img_metas" in data and data.get("img_metas") is not None:
                if hasattr(data["img_metas"], "data"):
                    dc_data = data["img_metas"].data
                    if len(dc_data) > 0:
                        img_metas_extracted = dc_data[0]
                elif isinstance(data["img_metas"], list) and len(data["img_metas"]) > 0:
                    if hasattr(data["img_metas"][0], "data"):
                        img_metas_extracted = (
                            data["img_metas"][0].data[0] if len(data["img_metas"][0].data) > 0 else None
                        )
                    else:
                        img_metas_extracted = data["img_metas"][0]

            img_metas = img_metas_extracted

            if "gt_bboxes_3d" in data and hasattr(data["gt_bboxes_3d"], "data"):
                gt_bboxes_3d = data["gt_bboxes_3d"].data[0] if len(data["gt_bboxes_3d"].data) > 0 else None
            else:
                gt_bboxes_3d = None

            if "gt_labels_3d" in data and hasattr(data["gt_labels_3d"], "data"):
                gt_labels_3d = data["gt_labels_3d"].data[0] if len(data["gt_labels_3d"].data) > 0 else None
            else:
                gt_labels_3d = None

            pts_filename = None
            if img_metas is not None and len(img_metas) > 0 and isinstance(img_metas[0], dict):
                pts_filename = img_metas[0].get("pts_filename", img_metas[0].get("lidar_path", ""))

            if not pts_filename:
                if "pts_filename" in data:
                    if hasattr(data["pts_filename"], "data"):
                        pts_filename_raw = data["pts_filename"].data[0] if len(data["pts_filename"].data) > 0 else None
                    else:
                        pts_filename_raw = data["pts_filename"]
                    if isinstance(pts_filename_raw, list) and len(pts_filename_raw) > 0:
                        pts_filename = pts_filename_raw[0]
                    elif isinstance(pts_filename_raw, str):
                        pts_filename = pts_filename_raw

            if not pts_filename:
                if "lidar_path" in data:
                    if hasattr(data["lidar_path"], "data"):
                        lidar_path_raw = data["lidar_path"].data[0] if len(data["lidar_path"].data) > 0 else None
                    else:
                        lidar_path_raw = data["lidar_path"]
                    if isinstance(lidar_path_raw, list) and len(lidar_path_raw) > 0:
                        pts_filename = lidar_path_raw[0]
                    elif isinstance(lidar_path_raw, str):
                        pts_filename = lidar_path_raw

            if not pts_filename or not isinstance(pts_filename, str):
                mmlogger.error(f"\n Cannot determine pts_filename for index {i}, got: {type(pts_filename)}, skipping")
                continue

            pts_filename = osp.basename(pts_filename)
            pts_filename_processed = pts_filename.replace("__LIDAR_TOP__", "_").split(".")[0]

            if len(CANDIDATE) > 0:
                normalized_candidates = []
                for candidate in CANDIDATE:
                    if "__CAM_" in candidate:
                        sample_token = candidate.split("__")[0]
                        normalized_candidates.append(sample_token)
                    else:
                        normalized_candidate = candidate.replace("__LIDAR_TOP__", "_").split(".")[0]
                        normalized_candidates.append(normalized_candidate)

                sample_token = (
                    pts_filename_processed.split("_")[0] if "_" in pts_filename_processed else pts_filename_processed
                )

                if sample_token not in [c.split("_")[0] if "_" in c else c for c in normalized_candidates]:
                    if pts_filename_processed not in normalized_candidates:
                        mmlogger.debug(f"Skipping sample {pts_filename_processed} - not in CANDIDATE list")
                        continue

            pts_filename = pts_filename_processed

            mmlogger.info(f"=== Processing sample {pts_filename} with TTNN model ===")
            mmlogger.info(f"Data keys: {list(data.keys())}")

            def extract_data_value(data, key):
                if key not in data:
                    return None
                val = data[key]
                if hasattr(val, "data"):
                    val = val.data[0] if len(val.data) > 0 else None
                return val

            # Construct img_metas dict from data fields
            constructed_img_metas = {}

            # Required fields for TtMapTR
            meta_keys = [
                "scene_token",
                "can_bus",
                "lidar2img",
                "filename",
                "img_shape",
                "ori_shape",
                "pad_shape",
                "scale_factor",
                "sample_idx",
                "pts_filename",
                "lidar_path",
                "prev_idx",
                "next_idx",
            ]

            for key in meta_keys:
                val = extract_data_value(data, key)
                if val is not None:
                    if key == "lidar2img":
                        constructed_img_metas[key] = val
                    elif key in ["img_shape", "ori_shape", "pad_shape"]:
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            if isinstance(val[0], (list, tuple)) and len(val[0]) >= 2:
                                constructed_img_metas[key] = val
                            elif isinstance(val[0], (int, float)):
                                constructed_img_metas[key] = [tuple(val)] * 6
                            else:
                                constructed_img_metas[key] = val
                        else:
                            constructed_img_metas[key] = val
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        if isinstance(val[0], (np.ndarray, str, int, float)):
                            constructed_img_metas[key] = val[0]
                        else:
                            constructed_img_metas[key] = val
                    else:
                        constructed_img_metas[key] = val

            # Ensure can_bus is a numpy array
            if "can_bus" in constructed_img_metas:
                can_bus_val = constructed_img_metas["can_bus"]
                if hasattr(can_bus_val, "numpy"):
                    constructed_img_metas["can_bus"] = can_bus_val.numpy()
                elif isinstance(can_bus_val, list):
                    constructed_img_metas["can_bus"] = np.array(can_bus_val)

            for shape_key in ["img_shape", "ori_shape", "pad_shape"]:
                if shape_key in constructed_img_metas:
                    shape_val = constructed_img_metas[shape_key]
                    mmlogger.info(f"{shape_key} raw: {shape_val}")

                    if isinstance(shape_val, np.ndarray):
                        if shape_val.ndim == 1 and len(shape_val) >= 2:
                            constructed_img_metas[shape_key] = [tuple(shape_val.tolist())] * 6
                        elif shape_val.ndim == 2:
                            constructed_img_metas[shape_key] = [tuple(row) for row in shape_val]
                    elif isinstance(shape_val, (list, tuple)):
                        if len(shape_val) > 0:
                            if isinstance(shape_val[0], (int, float, np.integer, np.floating)):
                                constructed_img_metas[shape_key] = [tuple(shape_val)] * 6
                            elif not isinstance(shape_val[0], (list, tuple)):
                                constructed_img_metas[shape_key] = [tuple(shape_val)] * 6

                    mmlogger.info(f"{shape_key} processed: {constructed_img_metas.get(shape_key)}")

            if "lidar2img" in constructed_img_metas:
                lidar2img_val = constructed_img_metas["lidar2img"]
                mmlogger.info(f"Raw lidar2img type: {type(lidar2img_val)}")
                if hasattr(lidar2img_val, "shape"):
                    mmlogger.info(f"Raw lidar2img shape: {lidar2img_val.shape}")
                elif hasattr(lidar2img_val, "__len__"):
                    mmlogger.info(f"Raw lidar2img len: {len(lidar2img_val)}")

                while isinstance(lidar2img_val, (list, tuple)) and len(lidar2img_val) == 1:
                    inner = lidar2img_val[0]
                    if isinstance(inner, (list, tuple)) and len(inner) > 1:
                        lidar2img_val = inner
                        break
                    elif isinstance(inner, np.ndarray) and inner.ndim >= 3:
                        lidar2img_val = inner
                        break
                    else:
                        break

                if isinstance(lidar2img_val, (list, tuple)):
                    if hasattr(lidar2img_val[0], "numpy"):
                        matrices = [m.numpy() for m in lidar2img_val]
                    elif isinstance(lidar2img_val[0], np.ndarray):
                        matrices = list(lidar2img_val)
                    else:
                        matrices = [np.array(m) for m in lidar2img_val]
                    lidar2img_val = np.stack(matrices)
                elif hasattr(lidar2img_val, "numpy"):
                    lidar2img_val = lidar2img_val.numpy()

                if isinstance(lidar2img_val, np.ndarray):
                    mmlogger.info(f"lidar2img after conversion shape: {lidar2img_val.shape}")

                    while lidar2img_val.ndim > 3:
                        # Find singleton dimensions to squeeze
                        squeezed = False
                        for dim in range(lidar2img_val.ndim):
                            if lidar2img_val.shape[dim] == 1:
                                lidar2img_val = np.squeeze(lidar2img_val, axis=dim)
                                mmlogger.info(f"Squeezed dim {dim}, new shape: {lidar2img_val.shape}")
                                squeezed = True
                                break
                        if not squeezed:
                            mmlogger.warning(f"Cannot squeeze lidar2img to 3D, shape: {lidar2img_val.shape}")
                            break

                    if lidar2img_val.ndim == 3 and lidar2img_val.shape[1:] == (4, 4):
                        constructed_img_metas["lidar2img"] = lidar2img_val
                        mmlogger.info(f"Final lidar2img shape: {lidar2img_val.shape}")
                    elif lidar2img_val.ndim == 2 and lidar2img_val.shape == (4, 4):
                        constructed_img_metas["lidar2img"] = np.stack([lidar2img_val] * 6)
                        mmlogger.info(f"Final lidar2img shape (duplicated): {constructed_img_metas['lidar2img'].shape}")
                    else:
                        mmlogger.warning(f"Unexpected final lidar2img shape: {lidar2img_val.shape}, using as-is")
                        constructed_img_metas["lidar2img"] = lidar2img_val
                else:
                    mmlogger.warning(f"lidar2img is not numpy array: {type(lidar2img_val)}")

            if img_metas is not None and len(img_metas) > 0 and isinstance(img_metas[0], dict):
                for key, val in constructed_img_metas.items():
                    if key not in img_metas[0]:
                        img_metas[0][key] = val
                final_img_metas = [[img_metas[0]]]
            else:
                final_img_metas = [[constructed_img_metas]]

            mmlogger.info(f"Constructed img_metas keys: {list(final_img_metas[0][0].keys())}")

            img_tensor = None
            if "img" in data:
                img_data = data["img"]
                mmlogger.info(f"=== DEBUG: Image Data Structure ===")
                mmlogger.info(f"data['img'] type: {type(img_data)}")

                if isinstance(img_data, (list, tuple)) and len(img_data) == 6:
                    mmlogger.info(f"data['img'] is list of {len(img_data)} camera images")
                    first_elem = img_data[0]
                    if hasattr(first_elem, "shape"):
                        mmlogger.info(f"Each camera image shape: {first_elem.shape}")
                        img_tensor = torch.stack(list(img_data), dim=0)
                        mmlogger.info(f"Stacked images shape: {img_tensor.shape}")

                        if img_tensor.dim() == 5:
                            img_tensor = img_tensor.squeeze(1)
                            mmlogger.info(f"After squeeze(1): {img_tensor.shape}")

                        if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
                            img_tensor = img_tensor.permute(0, 3, 1, 2)
                            mmlogger.info(f"After permute to (N, C, H, W): {img_tensor.shape}")
                elif isinstance(img_data, (list, tuple)) and len(img_data) > 0:
                    mmlogger.info(f"data['img'] is list/tuple of length: {len(img_data)}")
                    if hasattr(img_data[0], "data"):
                        inner_tensor = img_data[0].data[0] if len(img_data[0].data) > 0 else None
                        if inner_tensor is not None:
                            mmlogger.info(f"DataContainer inner tensor shape: {inner_tensor.shape}")
                            img_tensor = inner_tensor
                    elif hasattr(img_data[0], "shape"):
                        mmlogger.info(f"data['img'][0] is tensor with shape: {img_data[0].shape}")
                        img_tensor = img_data[0]
                elif hasattr(img_data, "data"):
                    mmlogger.info(f"data['img'] is DataContainer")
                    if len(img_data.data) > 0:
                        inner_tensor = img_data.data[0]
                        mmlogger.info(f"DataContainer inner tensor shape: {inner_tensor.shape}")
                        img_tensor = inner_tensor
                elif hasattr(img_data, "shape"):
                    mmlogger.info(f"data['img'] is tensor with shape: {img_data.shape}")
                    img_tensor = img_data

            if img_tensor is not None:
                mmlogger.info(
                    f"Raw image tensor shape: {img_tensor.shape if hasattr(img_tensor, 'shape') else type(img_tensor)}"
                )

                if len(img_tensor.shape) == 3:
                    H, W, C = img_tensor.shape
                    mmlogger.warning(f"Single camera image detected ({H}x{W}), duplicating for 6 cameras")
                    img_tensor = img_tensor.permute(2, 0, 1)
                    img_tensor = img_tensor.unsqueeze(0).repeat(6, 1, 1, 1)
                elif len(img_tensor.shape) == 4:
                    N, dim1, dim2, dim3 = img_tensor.shape
                    if dim3 == 3:
                        img_tensor = img_tensor.permute(0, 3, 1, 2)
                elif len(img_tensor.shape) == 5:
                    B, N, dim1, dim2, dim3 = img_tensor.shape
                    if dim3 == 3:
                        img_tensor = img_tensor.permute(0, 1, 4, 2, 3)
                    img_tensor = img_tensor.squeeze(0)

                _, _, curr_h, curr_w = img_tensor.shape
                resize_scale_h = 1.0
                resize_scale_w = 1.0
                if curr_h != expected_img_h or curr_w != expected_img_w:
                    mmlogger.info(f"Resizing images from {curr_h}x{curr_w} to {expected_img_h}x{expected_img_w}")
                    resize_scale_h = expected_img_h / curr_h
                    resize_scale_w = expected_img_w / curr_w
                    mmlogger.info(f"Resize scale factors: H={resize_scale_h:.4f}, W={resize_scale_w:.4f}")
                    import torch.nn.functional as F

                    img_tensor = F.interpolate(
                        img_tensor,
                        size=(expected_img_h, expected_img_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                    if "lidar2img" in final_img_metas[0][0]:
                        lidar2img_orig = final_img_metas[0][0]["lidar2img"]
                        if isinstance(lidar2img_orig, np.ndarray):
                            lidar2img_scaled = lidar2img_orig.copy()
                            lidar2img_scaled[:, 0, :] *= resize_scale_w
                            lidar2img_scaled[:, 1, :] *= resize_scale_h
                            final_img_metas[0][0]["lidar2img"] = lidar2img_scaled
                            mmlogger.info(f"Scaled lidar2img matrices for resize: shape {lidar2img_scaled.shape}")
                        else:
                            mmlogger.warning(f"Could not scale lidar2img - unexpected type: {type(lidar2img_orig)}")

                    final_img_metas[0][0]["img_shape"] = [(expected_img_h, expected_img_w, 3)] * 6
                    mmlogger.info(f"Updated img_shape to: {final_img_metas[0][0]['img_shape'][0]}")

                mmlogger.info(f"Processed image tensor shape: {img_tensor.shape}")

                img_tt = ttnn.from_torch(
                    img_tensor.float(),
                    dtype=ttnn.bfloat16,
                    device=device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                img_tt = [[img_tt]]
            else:
                img_tt = None
                mmlogger.warning("No image tensor found in data")

            with torch.no_grad():
                result = tt_model(
                    return_loss=False,
                    img=img_tt,
                    img_metas=final_img_metas,
                )

            mmlogger.info(f"=== DEBUG: TTNN Model Output Analysis ===")
            mmlogger.info(
                f"Result type: {type(result)}, length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}"
            )
            if isinstance(result, (list, tuple)) and len(result) > 0:
                mmlogger.info(f"Result[0] keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'N/A'}")

            sample_dir = osp.join(args.show_dir, pts_filename)
            os.makedirs(osp.abspath(sample_dir), exist_ok=True)

            if img_metas is not None and len(img_metas) > 0 and isinstance(img_metas[0], dict):
                filename_list = img_metas[0].get("filename", [])
            else:
                if "filename" in data:
                    if hasattr(data["filename"], "data"):
                        filename_list = data["filename"].data[0] if len(data["filename"].data) > 0 else []
                    else:
                        filename_list = data["filename"] if isinstance(data["filename"], list) else []
                else:
                    filename_list = []

            img_path_dict = {}
            for filepath in filename_list:
                if isinstance(filepath, (tuple, list)):
                    filepath = filepath[0] if len(filepath) > 0 else None
                if not filepath or not isinstance(filepath, str):
                    continue
                if not osp.exists(filepath):
                    mmlogger.warning(f"Image file not found: {filepath}, skipping")
                    continue
                try:
                    filename = osp.basename(filepath)
                    filename_splits = filename.split("__")
                    if len(filename_splits) < 2:
                        if "__" in filename:
                            parts = filename.split("__")
                            cam_name = parts[-1].split(".")[0] if "." in parts[-1] else parts[-1]
                        else:
                            cam_name = osp.basename(osp.dirname(filepath))
                        img_name = cam_name + ".jpg"
                    else:
                        img_name = filename_splits[1] + ".jpg"
                    img_path = osp.join(sample_dir, img_name)
                    shutil.copyfile(filepath, img_path)
                    cam_key = filename_splits[1] if len(filename_splits) >= 2 else cam_name
                    img_path_dict[cam_key] = img_path
                except Exception as e:
                    mmlogger.warning(f"Failed to process image {filepath}: {e}, skipping")
                    continue

            row_1_list = []
            for cam in CAMS[:3]:
                cam_img_name = cam + ".jpg"
                cam_img_path = osp.join(sample_dir, cam_img_name)
                if osp.exists(cam_img_path):
                    cam_img = cv2.imread(cam_img_path)
                    if cam_img is not None:
                        row_1_list.append(cam_img)

            row_2_list = []
            for cam in CAMS[3:]:
                cam_img_name = cam + ".jpg"
                cam_img_path = osp.join(sample_dir, cam_img_name)
                if osp.exists(cam_img_path):
                    cam_img = cv2.imread(cam_img_path)
                    if cam_img is not None:
                        row_2_list.append(cam_img)

            if len(row_1_list) > 0 or len(row_2_list) > 0:
                if len(row_1_list) > 0:
                    target_height = row_1_list[0].shape[0]
                    row_1_resized = []
                    for img_cv in row_1_list:
                        if img_cv.shape[0] != target_height:
                            img_cv = cv2.resize(
                                img_cv, (int(img_cv.shape[1] * target_height / img_cv.shape[0]), target_height)
                            )
                        row_1_resized.append(img_cv)
                    row_1_img = cv2.hconcat(row_1_resized) if len(row_1_resized) > 0 else None
                else:
                    row_1_img = None

                if len(row_2_list) > 0:
                    target_height = row_2_list[0].shape[0]
                    row_2_resized = []
                    for img_cv in row_2_list:
                        if img_cv.shape[0] != target_height:
                            img_cv = cv2.resize(
                                img_cv, (int(img_cv.shape[1] * target_height / img_cv.shape[0]), target_height)
                            )
                        row_2_resized.append(img_cv)
                    row_2_img = cv2.hconcat(row_2_resized) if len(row_2_resized) > 0 else None
                else:
                    row_2_img = None

                if row_1_img is not None and row_2_img is not None:
                    target_width = max(row_1_img.shape[1], row_2_img.shape[1])
                    if row_1_img.shape[1] != target_width:
                        row_1_img = cv2.resize(row_1_img, (target_width, row_1_img.shape[0]))
                    if row_2_img.shape[1] != target_width:
                        row_2_img = cv2.resize(row_2_img, (target_width, row_2_img.shape[0]))
                    cams_img = cv2.vconcat([row_1_img, row_2_img])
                elif row_1_img is not None:
                    cams_img = row_1_img
                elif row_2_img is not None:
                    cams_img = row_2_img
                else:
                    mmlogger.warning(f"No valid images found for surrounding view for sample {pts_filename}")
                    cams_img = None

                if cams_img is not None:
                    cams_img_path = osp.join(sample_dir, "surroud_view.jpg")
                    cv2.imwrite(cams_img_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 70])

            for vis_format in args.gt_format:
                if not has_gt or gt_bboxes_3d is None or gt_labels_3d is None:
                    mmlogger.warning(f"Skipping GT visualization for format {vis_format} - no GT data")
                    continue
                if vis_format == "se_pts":
                    gt_line_points = gt_bboxes_3d[0].start_end_points
                    for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                        pts = gt_bbox_3d.reshape(-1, 2).numpy()
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.quiver(
                            x[:-1],
                            y[:-1],
                            x[1:] - x[:-1],
                            y[1:] - y[:-1],
                            scale_units="xy",
                            angles="xy",
                            scale=1,
                            color=colors_plt[gt_label_3d],
                        )
                elif vis_format == "bbox":
                    gt_lines_bbox = gt_bboxes_3d[0].bbox
                    for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                        gt_bbox_3d = gt_bbox_3d.numpy()
                        xy = (gt_bbox_3d[0], gt_bbox_3d[1])
                        width = gt_bbox_3d[2] - gt_bbox_3d[0]
                        height = gt_bbox_3d[3] - gt_bbox_3d[1]
                        plt.gca().add_patch(
                            Rectangle(
                                xy, width, height, linewidth=0.4, edgecolor=colors_plt[gt_label_3d], facecolor="none"
                            )
                        )
                elif vis_format == "fixed_num_pts":
                    plt.figure(figsize=(2, 4))
                    plt.xlim(pc_range[0], pc_range[3])
                    plt.ylim(pc_range[1], pc_range[4])
                    plt.axis("off")
                    gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                    for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                        pts = gt_bbox_3d.numpy()
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.plot(x, y, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
                        plt.scatter(x, y, color=colors_plt[gt_label_3d], s=2, alpha=0.8, zorder=-1)
                    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                    gt_fixedpts_map_path = osp.join(sample_dir, "GT_fixednum_pts_MAP.png")
                    plt.savefig(gt_fixedpts_map_path, bbox_inches="tight", format="png", dpi=1200)
                    plt.close()
                elif vis_format == "polyline_pts":
                    plt.figure(figsize=(2, 4))
                    plt.xlim(pc_range[0], pc_range[3])
                    plt.ylim(pc_range[1], pc_range[4])
                    plt.axis("off")
                    gt_lines_instance = gt_bboxes_3d[0].instance_list
                    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                        pts = np.array(list(gt_line_instance.coords))
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.plot(x, y, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
                        plt.scatter(x, y, color=colors_plt[gt_label_3d], s=1, alpha=0.8, zorder=-1)
                    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                    gt_polyline_map_path = osp.join(sample_dir, "GT_polyline_pts_MAP.png")
                    plt.savefig(gt_polyline_map_path, bbox_inches="tight", format="png", dpi=1200)
                    plt.close()

                else:
                    mmlogger.error(f"WRONG visformat for GT: {vis_format}")
                    raise ValueError(f"WRONG visformat for GT: {vis_format}")

            plt.figure(figsize=(2, 4))
            plt.xlim(pc_range[0], pc_range[3])
            plt.ylim(pc_range[1], pc_range[4])
            plt.axis("off")

            result_dic = result[0]["pts_bbox"]
            boxes_3d = result_dic["boxes_3d"]
            scores_3d = result_dic["scores_3d"]
            labels_3d = result_dic["labels_3d"]
            pts_3d = result_dic["pts_3d"]

            mmlogger.info(f"=== DEBUG: Raw TTNN Model Output ===")
            mmlogger.info(
                f"boxes_3d type: {type(boxes_3d)}, shape: {boxes_3d.shape if hasattr(boxes_3d, 'shape') else 'N/A'}"
            )
            mmlogger.info(
                f"scores_3d type: {type(scores_3d)}, shape: {scores_3d.shape if hasattr(scores_3d, 'shape') else 'N/A'}"
            )
            mmlogger.info(
                f"labels_3d type: {type(labels_3d)}, shape: {labels_3d.shape if hasattr(labels_3d, 'shape') else 'N/A'}"
            )
            mmlogger.info(f"pts_3d type: {type(pts_3d)}, shape: {pts_3d.shape if hasattr(pts_3d, 'shape') else 'N/A'}")

            if hasattr(scores_3d, "cpu"):
                scores_3d = scores_3d.cpu().numpy()
            elif hasattr(scores_3d, "numpy"):
                scores_3d = scores_3d.numpy()
            elif not isinstance(scores_3d, np.ndarray):
                scores_3d = np.array(scores_3d)

            if hasattr(labels_3d, "cpu"):
                labels_3d = labels_3d.cpu().numpy()
            elif hasattr(labels_3d, "numpy"):
                labels_3d = labels_3d.numpy()
            elif not isinstance(labels_3d, np.ndarray):
                labels_3d = np.array(labels_3d)

            keep = scores_3d > args.score_thresh
            num_predictions = keep.sum()
            mmlogger.info(
                f"Found {num_predictions} predictions above threshold {args.score_thresh} (out of {len(scores_3d)} total)"
            )
            if len(scores_3d) > 0:
                mmlogger.info(f"All scores range: [{scores_3d.min():.4f}, {scores_3d.max():.4f}]")
                mmlogger.info(f"Unique labels: {np.unique(labels_3d)}")

            if num_predictions == 0:
                mmlogger.warning(f"No predictions above threshold {args.score_thresh} for sample {pts_filename}")
                if len(scores_3d) > 0:
                    max_score = float(scores_3d.max())
                    mmlogger.info(f"Max prediction score: {max_score:.4f}")

            plt.figure(figsize=(2, 4))
            plt.xlim(pc_range[0], pc_range[3])
            plt.ylim(pc_range[1], pc_range[4])
            plt.axis("off")

            pred_count = 0
            all_pred_x = []
            all_pred_y = []

            for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(
                scores_3d[keep], boxes_3d[keep], labels_3d[keep], pts_3d[keep]
            ):
                if hasattr(pred_pts_3d, "cpu"):
                    pred_pts_3d = pred_pts_3d.cpu().numpy()
                elif hasattr(pred_pts_3d, "numpy"):
                    pred_pts_3d = pred_pts_3d.numpy()
                elif not isinstance(pred_pts_3d, np.ndarray):
                    pred_pts_3d = np.array(pred_pts_3d)

                if pred_count < 3:
                    mmlogger.info(f"=== DEBUG: Prediction {pred_count} ===")
                    mmlogger.info(f"pred_pts_3d shape: {pred_pts_3d.shape}")
                    mmlogger.info(f"pred_pts_3d sample (first 5):\n{pred_pts_3d[:5]}")
                    mmlogger.info(f"pred_label_3d: {pred_label_3d}, pred_score_3d: {pred_score_3d:.4f}")

                pts_x = pred_pts_3d[:, 0]
                pts_y = pred_pts_3d[:, 1]

                all_pred_x.extend(pts_x)
                all_pred_y.extend(pts_y)

                if pred_count == 0:
                    mmlogger.info(
                        f"First prediction coordinate ranges - X: [{pts_x.min():.2f}, {pts_x.max():.2f}], Y: [{pts_y.min():.2f}, {pts_y.max():.2f}]"
                    )
                    mmlogger.info(
                        f"PC range: X: [{pc_range[0]:.2f}, {pc_range[3]:.2f}], Y: [{pc_range[1]:.2f}, {pc_range[4]:.2f}]"
                    )

                pred_label_idx = int(pred_label_3d) if hasattr(pred_label_3d, "__int__") else pred_label_3d
                if pred_label_idx < 0 or pred_label_idx >= len(colors_plt):
                    pred_label_idx = pred_label_idx % len(colors_plt) if pred_label_idx >= 0 else 0

                plt.plot(pts_x, pts_y, color=colors_plt[pred_label_idx], linewidth=1, alpha=0.8, zorder=-1)
                plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_idx], s=2, alpha=0.8, zorder=-1)
                pred_count += 1

            mmlogger.info(f"Visualized {pred_count} predictions for sample {pts_filename}")

            if has_gt and gt_bboxes_3d is not None and gt_labels_3d is not None and len(all_pred_x) > 0:
                try:
                    gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                    gt_all_x = []
                    gt_all_y = []
                    for gt_bbox_3d in gt_lines_fixed_num_pts:
                        pts = gt_bbox_3d.numpy()
                        gt_all_x.extend([pt[0] for pt in pts])
                        gt_all_y.extend([pt[1] for pt in pts])
                    if len(gt_all_x) > 0:
                        mmlogger.info(f"=== DEBUG: GT vs Pred Comparison ===")
                        mmlogger.info(
                            f"GT X: [{min(gt_all_x):.2f}, {max(gt_all_x):.2f}], Y: [{min(gt_all_y):.2f}, {max(gt_all_y):.2f}]"
                        )
                        mmlogger.info(
                            f"Pred X: [{min(all_pred_x):.2f}, {max(all_pred_x):.2f}], Y: [{min(all_pred_y):.2f}, {max(all_pred_y):.2f}]"
                        )
                        mmlogger.info(f"GT center: ({np.mean(gt_all_x):.2f}, {np.mean(gt_all_y):.2f})")
                        mmlogger.info(f"Pred center: ({np.mean(all_pred_x):.2f}, {np.mean(all_pred_y):.2f})")
                        mmlogger.info(
                            f"GT span: X={max(gt_all_x)-min(gt_all_x):.2f}, Y={max(gt_all_y)-min(gt_all_y):.2f}"
                        )
                        mmlogger.info(
                            f"Pred span: X={max(all_pred_x)-min(all_pred_x):.2f}, Y={max(all_pred_y)-min(all_pred_y):.2f}"
                        )
                except Exception as e:
                    mmlogger.debug(f"Could not compare with GT: {e}")

            plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

            map_path = osp.join(sample_dir, "PRED_MAP_plot.png")
            plt.savefig(map_path, bbox_inches="tight", format="png", dpi=1200)
            plt.close()

            prog_bar.update()

        mmlogger.info("\n DONE vis test dataset samples gt label & pred using TTNN model")

    finally:
        ttnn.close_device(device)
        mmlogger.info("TTNN device closed")


if __name__ == "__main__":
    main()
