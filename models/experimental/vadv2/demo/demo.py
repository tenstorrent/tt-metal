# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import copy
import time
import gc
import numpy as np
import pytest
import torch
import ttnn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS
from models.experimental.vadv2.reference import vad
from models.experimental.vadv2.common import load_torch_model
from models.experimental.vadv2.demo.nuscenes_vad_dataset import LiDARInstanceLines
import os.path as osp

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


def _env_flag_enabled(env_value, default=True):
    if env_value is None:
        return default
    return env_value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_VISUALIZATION = _env_flag_enabled(os.getenv("VADV2_ENABLE_VISUALIZATION"), default=True)
MEMORY_DEBUG = _env_flag_enabled(os.getenv("VADV2_MEMORY_DEBUG"), default=False)


def _get_memory_usage_mb():
    rss_mb = peak_mb = None
    if psutil is not None:
        process = psutil.Process(os.getpid())
        info = process.memory_info()
        rss_mb = info.rss / (1024**2)
    elif resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = usage.ru_maxrss
        if sys.platform.startswith("darwin"):
            rss_mb = rss / (1024**2)
        else:
            rss_mb = rss / 1024

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = usage.ru_maxrss
        if sys.platform.startswith("darwin"):
            peak_mb = peak / (1024**2)
        else:
            peak_mb = peak / 1024

    return rss_mb, peak_mb


def log_memory_usage(prefix):
    if not MEMORY_DEBUG:
        return

    rss_mb, peak_mb = _get_memory_usage_mb()
    if rss_mb is None and peak_mb is None:
        print(f"[Memory][{prefix}] Unable to determine memory usage (psutil/resource unavailable)")
        return

    if rss_mb is not None and peak_mb is not None:
        print(f"[Memory][{prefix}] RSS: {rss_mb:.2f} MB | Peak RSS: {peak_mb:.2f} MB")
    elif rss_mb is not None:
        print(f"[Memory][{prefix}] RSS: {rss_mb:.2f} MB")
    else:
        print(f"[Memory][{prefix}] Peak RSS: {peak_mb:.2f} MB")


# Register custom VAD transforms and datasets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import models.experimental.vadv2.demo.register_custom  # noqa: F401
from mmengine.utils import ProgressBar
from mmengine.dist import get_dist_info
from models.experimental.vadv2.tt.model_preprocessing import (
    create_vadv2_model_parameters_vad,
)
from models.experimental.vadv2.tt import tt_vad


@pytest.mark.parametrize("device_params", [{"l1_small_size": 20 * 1024}], indirect=True)
def test_tt_demo(device, model_location_generator):
    torch_model = vad.VAD(
        use_grid_mask=True,
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
        video_test_mode=True,
        fut_ts=6,
        fut_mode=6,
    )

    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "config.py"))
    # # 2. Build dataloader
    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    if MEMORY_DEBUG:
        log_memory_usage("tt demo - post dataloader init")
    # outputs = single_cpu_test(torch_model, dataloader)

    parameter = None
    for i, data in enumerate(dataloader):
        from models.experimental.vadv2.demo.data_container import DataContainer

        if MEMORY_DEBUG:
            log_memory_usage(f"tt demo - batch {i} raw")

        for k, v in data.items():
            if isinstance(v, DataContainer):
                v = v.data  # Unwrap DataContainer

            if isinstance(v, list):
                v = [item.data if isinstance(item, DataContainer) else item for item in v]
                v = [item.to("cpu") if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, torch.Tensor):
                v = v.to("cpu")

            data[k] = v

        img_for_params = data["img"][0] if isinstance(data["img"], list) else data["img"]

        if MEMORY_DEBUG:
            log_memory_usage(f"tt demo - before param build {i}")

        parameter = create_vadv2_model_parameters_vad(
            torch_model,
            img_for_params,
            device,
        )
        if MEMORY_DEBUG:
            log_memory_usage(f"tt demo - after param build {i}")
        break

    if parameter is None:
        raise RuntimeError("Failed to create VAD parameters for TT model initialization")

    tt_model = tt_vad.TtVAD(
        device,
        parameter,
        use_grid_mask=False,  # set to true for training only
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
        video_test_mode=True,
        fut_ts=6,
        fut_mode=6,
    )
    if MEMORY_DEBUG:
        log_memory_usage("tt demo - after TtVAD init")

    del parameter
    gc.collect()
    if MEMORY_DEBUG:
        log_memory_usage("tt demo - after parameter gc")

    if not ttnn.CONFIG.enable_model_cache:
        ttnn.CONFIG.enable_model_cache = True

    time_tag = time.strftime("%Y%m%d_%H%M%S")
    vis_dir = osp.join("test", "vad_tt_results", time_tag, "visualizations") if ENABLE_VISUALIZATION else None
    ttnn_outputs = single_cpu_test_tt(
        tt_model,
        dataloader,
        device,
        vis_dir=vis_dir,
        class_names=cfg.get("class_names"),
        max_vis_images=4 if ENABLE_VISUALIZATION else 0,
        score_thr=0.2,
        enable_visualization=ENABLE_VISUALIZATION,
    )
    if MEMORY_DEBUG:
        log_memory_usage("tt demo - after tt inference")

    # Add evaluation (same as test_torch_demo)
    tmp = {}
    tmp["bbox_results"] = ttnn_outputs
    ttnn_outputs = tmp
    rank, _ = get_dist_info()
    kwargs = {}
    kwargs["jsonfile_prefix"] = osp.join("test", "vad_tt_results", time_tag)
    eval_kwargs = cfg.get("evaluation", {}).copy()
    # hard-code way to remove EvalHook args
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox", **kwargs))

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  TT-Metal Evaluation outputs")
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataset_cfg = dataloader_cfg.pop("dataset")
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)

    print(dataset.evaluate(ttnn_outputs["bbox_results"], **eval_kwargs))


DEFAULT_COLOR_PALETTE = [
    (255, 69, 0),
    (30, 144, 255),
    (50, 205, 50),
    (255, 215, 0),
    (147, 112, 219),
    (255, 105, 180),
    (0, 206, 209),
    (255, 140, 0),
    (138, 43, 226),
    (0, 191, 255),
]

DEFAULT_GROUND_Z = -1.6
GT_LANE_PADDING_VALUE = -10000.0

BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _recover_images(img_tensor, img_norm_cfg):
    mean = np.array(img_norm_cfg.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.array(img_norm_cfg.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    to_rgb = img_norm_cfg.get("to_rgb", True)
    imgs = img_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    imgs = imgs * std.reshape(1, 1, 1, 3) + mean.reshape(1, 1, 1, 3)
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    if not to_rgb:
        imgs = imgs[..., ::-1]
    return imgs


def _project_corners_to_image(corners, lidar2img):
    corners_homo = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=corners.dtype)], axis=1)
    projections = corners_homo @ lidar2img.T
    depths = projections[:, 2]
    valid = depths > 1.0e-3
    if not np.all(valid):
        return None
    coords = projections[:, :2] / depths[:, None]
    return coords


def _project_polyline(points, lidar2img):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    if points.shape[1] == 2:
        zeros = np.zeros((points.shape[0], 1), dtype=points.dtype)
        points3d = np.concatenate([points, zeros], axis=1)
    else:
        points3d = points
    points_homo = np.concatenate([points3d, np.ones((points3d.shape[0], 1), dtype=points3d.dtype)], axis=1)
    projections = points_homo @ lidar2img.T
    depths = projections[:, 2]
    valid = depths > 1.0e-3
    if valid.sum() < 2:
        return None
    coords = projections[:, :2] / np.clip(depths[:, None], 1.0e-3, None)
    coords[~valid] = np.nan
    return coords


def _prepare_agent_trajectory(traj_entry, fut_ts=None):
    arr = np.asarray(traj_entry)
    if arr.size == 0:
        return None
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[0]
    elif arr.ndim == 4 and arr.shape[-1] == 2:
        arr = arr[0, 0]
    elif arr.ndim == 2 and arr.shape[-1] != 2:
        arr = arr.reshape(-1, 2)
    elif arr.ndim != 2:
        arr = arr.reshape(-1, 2)
    if arr.shape[-1] != 2:
        arr = arr.reshape(-1, 2)
    if fut_ts is not None and fut_ts > 0 and arr.shape[0] > fut_ts:
        arr = arr[:fut_ts]
    arr = np.cumsum(arr, axis=0)
    arr = np.vstack([np.zeros((1, 2), dtype=arr.dtype), arr])
    return arr


def _prepare_ego_trajectory(ego_preds, ego_cmd=None):
    if ego_preds is None:
        return None
    arr = np.asarray(ego_preds)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        mode_idx = 0
        if ego_cmd is not None:
            cmd = np.asarray(ego_cmd).reshape(-1)
            if cmd.size > 0:
                mode_idx = int(np.argmax(cmd))
                mode_idx = np.clip(mode_idx, 0, arr.shape[0] - 1)
        arr = arr[mode_idx]
    elif arr.ndim == 2 and arr.shape[-1] == 2:
        pass
    else:
        arr = arr.reshape(-1, 2)
    if arr.shape[-1] != 2:
        arr = arr.reshape(-1, 2)
    arr = np.cumsum(arr, axis=0)
    arr = np.vstack([np.zeros((1, 2), dtype=arr.dtype), arr])
    return arr


def _camera_name_from_path(path):
    parent = osp.basename(osp.dirname(path))
    return parent if parent else "camera"


def _extract_ground_truth_lanes(data_entry):
    gt_container = data_entry.get("map_gt_bboxes_3d")
    gt_labels = data_entry.get("map_gt_labels_3d")

    if gt_container is None or gt_labels is None:
        return None, None

    if hasattr(gt_container, "data"):
        gt_container = gt_container.data
    if hasattr(gt_labels, "data"):
        gt_labels = gt_labels.data

    if isinstance(gt_labels, torch.Tensor):
        gt_labels_np = gt_labels.detach().cpu().numpy()
    else:
        gt_labels_np = np.asarray(gt_labels)

    if gt_labels_np.size == 0:
        return None, None

    gt_pts_np = None
    if isinstance(gt_container, torch.Tensor):
        gt_pts_np = gt_container.detach().cpu().numpy()
    elif isinstance(gt_container, LiDARInstanceLines):
        if len(gt_container.instance_list) == 0:
            return None, None
        pts_tensor = gt_container.fixed_num_sampled_points_torch
        if isinstance(pts_tensor, torch.Tensor):
            gt_pts_np = pts_tensor.detach().cpu().numpy()
        else:
            gt_pts_np = np.asarray(pts_tensor)
    else:
        try:
            gt_pts_np = np.asarray(gt_container)
        except Exception:
            return None, None

    if gt_pts_np is None or gt_pts_np.size == 0:
        return None, None

    if not isinstance(gt_pts_np, np.ndarray):
        gt_pts_np = np.asarray(gt_pts_np)

    if gt_pts_np.ndim != 3:
        return None, None

    lane_count = min(gt_pts_np.shape[0], gt_labels_np.shape[0])
    if lane_count == 0:
        return None, None

    gt_pts_np = gt_pts_np[:lane_count]
    gt_labels_np = gt_labels_np[:lane_count]

    return gt_pts_np, gt_labels_np


def visualize_sample_prediction(data, result, out_dir, class_names, score_thr=0.2, color_palette=None):
    if "img" not in data or "img_metas" not in data:
        return
    if color_palette is None:
        color_palette = DEFAULT_COLOR_PALETTE

    img_batch = data["img"]
    img_metas_batch = data["img_metas"]
    if not isinstance(img_batch, list) or len(img_batch) == 0:
        return

    img_tensor = img_batch[0]
    if not isinstance(img_tensor, torch.Tensor):
        return

    img_metas = img_metas_batch[0] if isinstance(img_metas_batch, list) and img_metas_batch else {}
    img_norm_cfg = img_metas.get("img_norm_cfg", {})
    recovered_imgs = _recover_images(img_tensor, img_norm_cfg)

    pred_dict = result.get("pts_bbox", result if isinstance(result, dict) else {})
    boxes_3d = pred_dict.get("boxes_3d")
    scores_3d = pred_dict.get("scores_3d")
    labels_3d = pred_dict.get("labels_3d")
    lane_pts = pred_dict.get("map_pts_3d")
    lane_scores = pred_dict.get("map_scores_3d")
    lane_labels = pred_dict.get("map_labels_3d")
    if boxes_3d is None or scores_3d is None or labels_3d is None:
        return

    boxes_3d = boxes_3d.to("cpu")
    box_tensor = boxes_3d.tensor.cpu().numpy()
    corners_3d = boxes_3d.corners.cpu().numpy()
    scores_3d = scores_3d.detach().cpu().numpy()
    labels_3d = labels_3d.detach().cpu().numpy()

    valid_mask = scores_3d >= score_thr
    if not np.any(valid_mask):
        valid_mask = np.ones_like(scores_3d, dtype=bool)

    corners_3d = corners_3d[valid_mask]
    scores_3d = scores_3d[valid_mask]
    labels_3d = labels_3d[valid_mask]
    box_tensor = box_tensor[valid_mask]
    if corners_3d.size > 0:
        ground_z = float(np.min(corners_3d[..., 2]))
    else:
        ground_z = DEFAULT_GROUND_Z

    lane_pts_np = None
    lane_labels_np = None
    if lane_pts is not None and lane_scores is not None and lane_labels is not None:
        lane_scores_np = np.asarray(lane_scores.detach().cpu().numpy())
        lane_mask = lane_scores_np >= score_thr
        if np.any(lane_mask):
            lane_pts_np = np.asarray(lane_pts.detach().cpu().numpy())[lane_mask]
            lane_labels_np = np.asarray(lane_labels.detach().cpu().numpy())[lane_mask]
            if lane_pts_np.ndim == 3 and lane_pts_np.shape[-1] == 2:
                zeros = np.zeros((*lane_pts_np.shape[:-1], 1), dtype=lane_pts_np.dtype)
                lane_pts_np = np.concatenate([lane_pts_np, zeros], axis=-1)
            if lane_pts_np.ndim == 3 and lane_pts_np.shape[-1] == 3:
                lane_pts_np[..., 2] = ground_z
        else:
            lane_pts_np = None

    agent_trajs_np = None
    raw_agent_trajs = pred_dict.get("trajs_3d")
    if raw_agent_trajs is not None:
        if isinstance(raw_agent_trajs, torch.Tensor):
            agent_trajs_np = raw_agent_trajs.detach().cpu().numpy()
        else:
            agent_trajs_np = np.asarray(raw_agent_trajs)
        if agent_trajs_np.shape[0] == valid_mask.shape[0]:
            agent_trajs_np = agent_trajs_np[valid_mask]
        else:
            agent_trajs_np = None

    ego_preds = pred_dict.get("ego_fut_preds")
    ego_cmd = pred_dict.get("ego_fut_cmd")
    ego_traj_xy = None
    if ego_preds is not None:
        ego_np = ego_preds.detach().cpu().numpy() if isinstance(ego_preds, torch.Tensor) else np.asarray(ego_preds)
        ego_cmd_np = (
            (ego_cmd.detach().cpu().numpy() if isinstance(ego_cmd, torch.Tensor) else np.asarray(ego_cmd))
            if ego_cmd is not None
            else None
        )
        ego_traj_xy = _prepare_ego_trajectory(ego_np, ego_cmd_np)
    fut_ts = ego_traj_xy.shape[0] - 1 if ego_traj_xy is not None else None

    lidar2img_list = img_metas.get("lidar2img", [])
    filenames = img_metas.get("filename", [])
    img_shapes = img_metas.get("img_shape", [])

    sample_token = (
        img_metas.get("sample_idx")
        or img_metas.get("sample_token")
        or img_metas.get("frame_id")
        or img_metas.get("timestamp")
        or "sample"
    )
    sample_token = str(sample_token)

    gt_lane_pts_np, gt_lane_labels_np = _extract_ground_truth_lanes(data)

    os.makedirs(out_dir, exist_ok=True)

    for cam_idx in range(recovered_imgs.shape[0]):
        if cam_idx >= len(lidar2img_list):
            break

        lidar2img = np.asarray(lidar2img_list[cam_idx])
        image_array = recovered_imgs[cam_idx]
        height, width = image_array.shape[:2]
        if cam_idx < len(img_shapes):
            height, width = img_shapes[cam_idx][:2]
            image_array = image_array[:height, :width]

        fig_w = max(width / 200.0, 4.0)
        fig_h = max(height / 200.0, 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(image_array)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        for idx, corners in enumerate(corners_3d):
            projected = _project_corners_to_image(corners, lidar2img)
            if projected is None:
                continue

            palette_index = int(labels_3d[idx]) % len(color_palette) if len(color_palette) > 0 else 0
            color_rgb = np.array(color_palette[palette_index], dtype=np.float32) / 255.0
            color_tuple = tuple(color_rgb.tolist())

            for edge in BOX_EDGES:
                pts = projected[list(edge)]
                ax.plot(pts[:, 0], pts[:, 1], color=color_tuple, linewidth=1.5)

            label_name = class_names[int(labels_3d[idx])] if class_names else str(labels_3d[idx])
            label_position = projected.mean(axis=0)
            ax.text(
                label_position[0],
                label_position[1],
                f"{label_name} {scores_3d[idx]:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color_tuple + (0.4,), edgecolor="none", pad=0.4),
            )

            if agent_trajs_np is not None and idx < agent_trajs_np.shape[0]:
                agent_rel = _prepare_agent_trajectory(agent_trajs_np[idx], fut_ts)
                if agent_rel is not None and agent_rel.shape[0] >= 2:
                    center_xy = box_tensor[idx, :2]
                    agent_world_xy = agent_rel + center_xy
                    agent_points3d = np.column_stack(
                        [agent_world_xy, np.full((agent_world_xy.shape[0]), ground_z, dtype=agent_world_xy.dtype)]
                    )
                    projected_agent = _project_polyline(agent_points3d, lidar2img)
                    if projected_agent is not None:
                        ax.plot(
                            projected_agent[:, 0],
                            projected_agent[:, 1],
                            color=color_tuple,
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.9,
                        )

        if lane_pts_np is not None:
            for lane_idx, lane in enumerate(lane_pts_np):
                projected_lane = _project_polyline(lane, lidar2img)
                if projected_lane is None:
                    continue
                lane_palette_idx = int(lane_labels_np[lane_idx]) % len(color_palette) if len(color_palette) > 0 else 0
                lane_color_rgb = np.array(color_palette[lane_palette_idx], dtype=np.float32) / 255.0
                lane_color = tuple(lane_color_rgb.tolist())
                ax.plot(
                    projected_lane[:, 0],
                    projected_lane[:, 1],
                    color=lane_color,
                    linewidth=2.0,
                    alpha=0.8,
                )

        if gt_lane_pts_np is not None and gt_lane_labels_np is not None:
            for gt_idx, gt_lane in enumerate(gt_lane_pts_np):
                if gt_lane.ndim != 2 or gt_lane.shape[0] < 2:
                    continue

                if gt_lane.shape[1] >= 3:
                    lane_xy = gt_lane[:, :2]
                elif gt_lane.shape[1] == 2:
                    lane_xy = gt_lane
                else:
                    continue

                valid_mask = ~np.any(np.isclose(lane_xy, GT_LANE_PADDING_VALUE), axis=1)
                lane_xy = lane_xy[valid_mask]
                if lane_xy.shape[0] < 2:
                    continue

                finite_mask = np.all(np.isfinite(lane_xy), axis=1)
                lane_xy = lane_xy[finite_mask]
                if lane_xy.shape[0] < 2:
                    continue

                lane_points3d = np.column_stack([lane_xy, np.full((lane_xy.shape[0]), ground_z, dtype=lane_xy.dtype)])
                projected_gt_lane = _project_polyline(lane_points3d, lidar2img)
                if projected_gt_lane is None:
                    continue

                gt_palette_idx = int(gt_lane_labels_np[gt_idx]) % len(color_palette) if len(color_palette) > 0 else 0
                gt_color_rgb = np.array(color_palette[gt_palette_idx], dtype=np.float32) / 255.0
                gt_color = tuple(gt_color_rgb.tolist())

                ax.plot(
                    projected_gt_lane[:, 0],
                    projected_gt_lane[:, 1],
                    color=gt_color,
                    linewidth=1.6,
                    linestyle=":",
                    alpha=0.9,
                )

        if ego_traj_xy is not None and ego_traj_xy.shape[0] >= 2:
            ego_points3d = np.column_stack(
                [ego_traj_xy, np.full((ego_traj_xy.shape[0]), ground_z, dtype=ego_traj_xy.dtype)]
            )
            projected_ego = _project_polyline(ego_points3d, lidar2img)
            if projected_ego is not None:
                ax.plot(
                    projected_ego[:, 0],
                    projected_ego[:, 1],
                    color=(1.0, 0.0, 1.0),
                    linewidth=2.5,
                    alpha=0.9,
                )

        ax.axis("off")
        fig.tight_layout(pad=0)
        camera_name = _camera_name_from_path(filenames[cam_idx]) if cam_idx < len(filenames) else f"camera_{cam_idx}"
        output_path = osp.join(out_dir, f"{sample_token}_{camera_name}.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def single_cpu_test_tt(
    tt_model,
    dataloader,
    device,
    vis_dir=None,
    class_names=None,
    max_vis_images=0,
    score_thr=0.2,
    enable_visualization=True,
):
    results = []
    prog_bar = ProgressBar(81)
    vis_count = 0
    for i, data in enumerate(dataloader):
        from models.experimental.vadv2.demo.data_container import DataContainer

        if MEMORY_DEBUG:
            log_memory_usage(f"tt inference iter {i} - start")

        for k, v in data.items():
            if isinstance(v, DataContainer):
                v = v.data  # Unwrap DataContainer

            if isinstance(v, list):
                v = [item.data if isinstance(item, DataContainer) else item for item in v]
                v = [item.to("cpu") if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, torch.Tensor):
                v = v.to("cpu")

            data[k] = v

        img_tensor = data["img"][0]
        if MEMORY_DEBUG:
            log_memory_usage(f"tt inference iter {i} - pre to_ttnn")

        tt_img = ttnn.from_torch(img_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        if MEMORY_DEBUG:
            log_memory_usage(f"tt inference iter {i} - post to_ttnn")

        with torch.no_grad():
            result = tt_model(
                return_loss=False,
                img=[[tt_img]],
                img_metas=[[data["img_metas"]]],
                gt_bboxes_3d=[[data["gt_bboxes_3d"]]],
                gt_labels_3d=[[data["gt_labels_3d"]]],
                fut_valid_flag=[data["fut_valid_flag"]],
                ego_his_trajs=[data["ego_his_trajs"]],
                ego_fut_trajs=[[data["ego_fut_trajs"][0].unsqueeze(0)]],
                ego_fut_cmd=[[data["ego_fut_cmd"][0].unsqueeze(0)]],
                ego_lcf_feat=[data["ego_lcf_feat"]],
                gt_attr_labels=[[data["gt_attr_labels"]]],
            )
        if MEMORY_DEBUG:
            log_memory_usage(f"tt inference iter {i} - post inference")

        del tt_img
        del img_tensor
        results.extend(result)

        if MEMORY_DEBUG:
            log_memory_usage(f"tt inference iter {i} - post extend")

        if enable_visualization and vis_dir and vis_count < max_vis_images:
            os.makedirs(vis_dir, exist_ok=True)
            for result_item in result:
                if vis_count >= max_vis_images:
                    break
                visualize_sample_prediction(
                    data,
                    result_item,
                    vis_dir,
                    class_names,
                    score_thr=score_thr,
                )
                vis_count += 1

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

        if MEMORY_DEBUG:
            log_memory_usage(f"tt inference iter {i} - end")

    return results


def test_torch_demo(model_location_generator):
    torch_model = vad.VAD(
        use_grid_mask=True,
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
        video_test_mode=True,
        fut_ts=6,
        fut_mode=6,
    )

    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "config.py"))
    # # 2. Build dataloader
    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    if MEMORY_DEBUG:
        log_memory_usage("torch demo - post dataloader init")
    time_tag = time.strftime("%Y%m%d_%H%M%S")
    vis_dir = osp.join("test", "vad_torch_results", time_tag, "visualizations") if ENABLE_VISUALIZATION else None
    outputs = single_cpu_test(
        torch_model,
        dataloader,
        vis_dir=vis_dir,
        class_names=cfg.get("class_names"),
        max_vis_images=4 if ENABLE_VISUALIZATION else 0,
        score_thr=0.2,
        enable_visualization=ENABLE_VISUALIZATION,
    )

    tmp = {}
    # 3. evaluation part -  WIP
    tmp["bbox_results"] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    kwargs = {}
    kwargs["jsonfile_prefix"] = osp.join("test", "vad_results", time.ctime().replace(" ", "_").replace(":", "_"))
    eval_kwargs = cfg.get("evaluation", {}).copy()
    # hard-code way to remove EvalHook args
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox", **kwargs))

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  Evaluation outputs")
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataset_cfg = dataloader_cfg.pop("dataset")
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)

    print(dataset.evaluate(outputs["bbox_results"], **eval_kwargs))


def single_cpu_test(
    torch_model,
    dataloader,
    vis_dir=None,
    class_names=None,
    max_vis_images=0,
    score_thr=0.2,
    enable_visualization=True,
):
    results = []
    prog_bar = ProgressBar(81)
    vis_count = 0
    for i, data in enumerate(dataloader):
        from models.experimental.vadv2.demo.data_container import DataContainer

        if MEMORY_DEBUG:
            log_memory_usage(f"torch demo iter {i} - start")

        for k, v in data.items():
            if isinstance(v, DataContainer):
                v = v.data  # Unwrap DataContainer

            if isinstance(v, list):
                v = [item.data if isinstance(item, DataContainer) else item for item in v]
                v = [item.to("cpu") if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, torch.Tensor):
                v = v.to("cpu")

            data[k] = v
        print(data)
        with torch.no_grad():
            result = torch_model(
                return_loss=False,
                img=[data["img"]],
                img_metas=[[data["img_metas"]]],
                gt_bboxes_3d=[[data["gt_bboxes_3d"]]],
                gt_labels_3d=[[data["gt_labels_3d"]]],
                fut_valid_flag=[data["fut_valid_flag"]],
                ego_his_trajs=[data["ego_his_trajs"]],
                ego_fut_trajs=[[data["ego_fut_trajs"][0].unsqueeze(0)]],
                ego_fut_cmd=[[data["ego_fut_cmd"][0].unsqueeze(0)]],
                ego_lcf_feat=[data["ego_lcf_feat"]],
                gt_attr_labels=[[data["gt_attr_labels"]]],
            )
        if MEMORY_DEBUG:
            log_memory_usage(f"torch demo iter {i} - post inference")
        results.extend(result)

        if MEMORY_DEBUG:
            log_memory_usage(f"torch demo iter {i} - post extend")

        if enable_visualization and vis_dir and vis_count < max_vis_images:
            os.makedirs(vis_dir, exist_ok=True)
            for result_item in result:
                if vis_count >= max_vis_images:
                    break
                visualize_sample_prediction(
                    data,
                    result_item,
                    vis_dir,
                    class_names,
                    score_thr=score_thr,
                )
                vis_count += 1

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

        if MEMORY_DEBUG:
            log_memory_usage(f"torch demo iter {i} - end")

    return results
