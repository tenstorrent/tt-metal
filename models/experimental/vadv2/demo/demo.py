# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import copy
import time
import pytest
import ttnn
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS
from models.experimental.vadv2.reference import vad
from models.experimental.vadv2.common import load_torch_model
import os.path as osp
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

    cfg = Config.fromfile("/home/ubuntu/sabira/tt-metal/models/experimental/vadv2/demo/config.py")
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

    ttnn_outputs = single_cpu_test_tt(tt_model, dataloader, device)


def single_cpu_test_tt(tt_model, dataloader, device):
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

    cfg = Config.fromfile("/home/ubuntu/sabira/tt-metal/models/experimental/vadv2/demo/config.py")
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
