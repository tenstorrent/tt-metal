# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import copy
import time
import numpy as np
import pytest
import torch
import ttnn
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS
from models.experimental.vadv2.reference import vad
from models.experimental.vadv2.common import load_torch_model
import os.path as osp
from PIL import Image, ImageDraw

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
    # outputs = single_cpu_test(torch_model, dataloader)

    for i, data in enumerate(dataloader):
        from models.experimental.vadv2.demo.data_container import DataContainer

        for k, v in data.items():
            if isinstance(v, DataContainer):
                v = v.data  # Unwrap DataContainer

            if isinstance(v, list):
                v = [item.data if isinstance(item, DataContainer) else item for item in v]
                v = [item.to("cpu") if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, torch.Tensor):
                v = v.to("cpu")

            data[k] = v

        parameter = create_vadv2_model_parameters_vad(
            torch_model,
            [
                False,
                [data["img"]],
                [[data["img_metas"]]],
                [[data["gt_bboxes_3d"]]],
                [[data["gt_labels_3d"]]],
                [data["fut_valid_flag"]],
                [data["ego_his_trajs"]],
                [[data["ego_fut_trajs"][0].unsqueeze(0)]],
                [[data["ego_fut_cmd"][0].unsqueeze(0)]],
                [data["ego_lcf_feat"]],
                [[data["gt_attr_labels"]]],
            ],
            device,
        )
        if i == 2:
            break

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

    time_tag = time.strftime("%Y%m%d_%H%M%S")
    vis_dir = osp.join("test", "vad_tt_results", time_tag, "visualizations")
    ttnn_outputs = single_cpu_test_tt(
        tt_model,
        dataloader,
        device,
        vis_dir=vis_dir,
        class_names=cfg.get("class_names"),
        max_vis_images=4,
        score_thr=0.2,
    )

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

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  TT-Metal Evaluation outputs")
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


def _project_box_to_image(corners, lidar2img):
    num_pts = corners.shape[0]
    corners_homo = np.concatenate([corners, np.ones((num_pts, 1), dtype=corners.dtype)], axis=1)
    projections = corners_homo @ lidar2img.T
    depths = projections[:, 2]
    valid = depths > 1.0e-3
    if not np.any(valid):
        return None
    projections = projections[valid]
    depths = depths[valid][:, None]
    coords = projections[:, :2] / depths
    return coords


def _camera_name_from_path(path):
    parent = osp.basename(osp.dirname(path))
    return parent if parent else "camera"


def visualize_sample_prediction(data, result, out_dir, class_names, score_thr=0.2, color_palette=None):
    if "img" not in data or "img_metas" not in data:
        return
    if color_palette is None:
        color_palette = DEFAULT_COLOR_PALETTE

    os.makedirs(out_dir, exist_ok=True)
    img_batch = data["img"]
    img_metas_batch = data["img_metas"]
    if not isinstance(img_batch, list) or len(img_batch) == 0:
        return
    img_tensor = img_batch[0]
    if not isinstance(img_tensor, torch.Tensor):
        return
    img_metas = img_metas_batch[0]
    img_norm_cfg = img_metas.get("img_norm_cfg", {})
    recovered_imgs = _recover_images(img_tensor, img_norm_cfg)

    pred_dict = result.get("pts_bbox", result if isinstance(result, dict) else {})
    boxes_3d = pred_dict.get("boxes_3d", None)
    scores_3d = pred_dict.get("scores_3d", None)
    labels_3d = pred_dict.get("labels_3d", None)
    if boxes_3d is None or scores_3d is None or labels_3d is None:
        return

    boxes_3d = boxes_3d.to("cpu")
    scores_3d = scores_3d.detach().cpu().numpy()
    labels_3d = labels_3d.detach().cpu().numpy()
    corners_3d = boxes_3d.corners.cpu().numpy()

    valid_mask = scores_3d >= score_thr
    if not np.any(valid_mask):
        valid_mask = np.ones_like(scores_3d, dtype=bool)

    sample_token = img_metas.get("sample_idx", "sample")
    lidar2img_list = img_metas.get("lidar2img", [])
    filenames = img_metas.get("filename", [])
    img_shapes = img_metas.get("img_shape", [])

    for cam_idx in range(recovered_imgs.shape[0]):
        if cam_idx >= len(lidar2img_list):
            break
        lidar2img = np.asarray(lidar2img_list[cam_idx])
        image_array = recovered_imgs[cam_idx]
        height, width = image_array.shape[:2]
        if cam_idx < len(img_shapes):
            height, width = img_shapes[cam_idx][:2]
        image = Image.fromarray(image_array[:height, :width])
        draw = ImageDraw.Draw(image)

        for box_idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            coords = _project_box_to_image(corners_3d[box_idx], lidar2img)
            if coords is None or coords.size == 0:
                continue
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            if x_max < 0 or y_max < 0 or x_min > width or y_min > height:
                continue
            x_min = float(np.clip(x_min, 0, width - 1))
            y_min = float(np.clip(y_min, 0, height - 1))
            x_max = float(np.clip(x_max, 0, width - 1))
            y_max = float(np.clip(y_max, 0, height - 1))
            color = color_palette[int(labels_3d[box_idx]) % len(color_palette)]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            label_name = class_names[int(labels_3d[box_idx])] if class_names else str(labels_3d[box_idx])
            draw.text((x_min + 2, y_min + 2), f"{label_name}:{scores_3d[box_idx]:.2f}", fill=color)

        camera_name = _camera_name_from_path(filenames[cam_idx]) if cam_idx < len(filenames) else f"camera_{cam_idx}"
        output_path = osp.join(out_dir, f"{sample_token}_{camera_name}.png")
        image.save(output_path)


def single_cpu_test_tt(
    tt_model,
    dataloader,
    device,
    vis_dir=None,
    class_names=None,
    max_vis_images=0,
    score_thr=0.2,
):
    results = []
    prog_bar = ProgressBar(81)
    vis_count = 0
    for i, data in enumerate(dataloader):
        from models.experimental.vadv2.demo.data_container import DataContainer

        for k, v in data.items():
            if isinstance(v, DataContainer):
                v = v.data  # Unwrap DataContainer

            if isinstance(v, list):
                v = [item.data if isinstance(item, DataContainer) else item for item in v]
                v = [item.to("cpu") if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, torch.Tensor):
                v = v.to("cpu")

            data[k] = v

        with torch.no_grad():
            result = tt_model(
                return_loss=False,
                img=[
                    [ttnn.from_torch(data["img"][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)]
                ],
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
        results.extend(result)

        if vis_dir and vis_count < max_vis_images:
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

        print(
            "==================================================one iteration is doneee ============================================================"
        )

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
    outputs = single_cpu_test(torch_model, dataloader)

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


def single_cpu_test(torch_model, dataloader):
    results = []
    prog_bar = ProgressBar(81)
    for i, data in enumerate(dataloader):
        from models.experimental.vadv2.demo.data_container import DataContainer

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
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results
