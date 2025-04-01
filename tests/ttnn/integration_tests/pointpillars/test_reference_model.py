# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.functional_pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN


def test_reference():
    reference_model = MVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
    )
    # print("reference_model",reference_model)
    state_dict = torch.load(
        "/home/ubuntu/pointpillars_mmdetect/mmdetection3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
    )["state_dict"]
    # print("state_dict",state_dict)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()
    batch_inputs_dict = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/functional_pointpillars/reference/batch_inputs_dict.pt"
    )
    batch_data_samples_modified = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/functional_pointpillars/reference/batch_inputs_metas_motdified.pt"
    )  # modified

    output = reference_model(batch_inputs_dict, batch_data_samples_modified)
