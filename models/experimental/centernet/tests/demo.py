# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_centernet import Ttcenternet
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.apis.det_inferencer import DetInferencer
from mmengine.config import Config
import numpy as np
from mmdet.apis.det_inferencer import DetInferencer
from rich.progress import track


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet(device, reset_seeds):
    inputs = "/home/ubuntu/sudharsan/tt-metal/models/experimental/centernet/demo.jpg"
    return_vis = False
    show = False
    wait_time = 0
    no_save_vis = False
    draw_pred = True
    pred_score_thr = 0.3
    print_result = False
    no_save_pred = False
    batch_size = 1
    show_progress = True
    tokens_positive = None

    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]

    base_inference = DetInferencer(
        model="models/experimental/centernet/tt/config.py",
        weights="models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth",
    )
    cfg = Config.fromfile("models/experimental/centernet/tt/config.py")

    pred_by_feat = CenterNetHead(
        in_channels=64,
        feat_channels=64,
        num_classes=80,
        test_cfg=cfg.model["test_cfg"],
        init_cfg=None,
    )
    pre_process = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=1,
        pad_value=0,
        pad_mask=False,
        mask_pad_value=0,
        pad_seg=False,
        seg_pad_value=255,
        bgr_to_rgb=True,
        rgb_to_bgr=False,
        boxtype2tensor=True,
        non_blocking=False,
        batch_augments=None,
    )
    (
        preprocess_kwargs,
        forward_kwargs,
        visualize_kwargs,
        postprocess_kwargs,
    ) = base_inference._dispatch_kwargs()

    ori_inputs = base_inference._inputs_to_list(inputs)
    tokens_positive = [tokens_positive] * len(ori_inputs)
    inputs = base_inference.preprocess(ori_inputs, batch_size, **preprocess_kwargs)

    parameters = custom_preprocessor(device, state_dict)
    centernet = Ttcenternet(parameters=parameters, device=device)
    results_dict = {"predictions": [], "visualization": []}

    for ori_imgs, data in track(inputs, description="Inference") if show_progress else inputs:
        data = pre_process(data, False)

        tt_input = ttnn.from_torch(data["inputs"][0].unsqueeze(0), dtype=ttnn.bfloat16, device=device)
        output = centernet.forward(tt_input)
        pred1 = ttnn.to_torch(output[0])
        pred2 = ttnn.to_torch(output[1])
        pred3 = ttnn.to_torch(output[2])
        batch_img_metas = [
            {
                "border": np.array([11.0, 438.0, 16.0, 656.0], dtype=np.float32),
                "pad_shape": (448, 672),
                "ori_shape": (427, 640),
                "batch_input_shape": (448, 672),
                "img_path": "models/experimental/centernet/demo.jpg",
                "img_shape": (448, 672),
            }
        ]
        preds = pred_by_feat.predict_by_feat(
            *([pred1], [pred2], [pred3]),
            batch_img_metas=batch_img_metas,
            rescale=True,
        )
        prediction = data["data_samples"]
        prediction[0].pred_instances = preds[0]

        visualization = base_inference.visualize(
            ori_imgs,
            prediction,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=True,
            pred_score_thr=pred_score_thr,
            no_save_vis=no_save_vis,
            img_out_dir="outputs",
            **visualize_kwargs,
        )

        results = base_inference.postprocess(
            prediction,
            visualization,
            return_datasamples=False,
            print_result=print_result,
            no_save_pred=no_save_pred,
            pred_out_dir="outputs",
            **postprocess_kwargs,
        )
        results_dict["predictions"].extend(results["predictions"])
        if results["visualization"] is not None:
            results_dict["visualization"].extend(results["visualization"])

        return results_dict
