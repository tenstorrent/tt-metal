# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from tests.ttnn.utils_for_testing import comp_pcc

from models.experimental.pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN


def test_reference():
    reference_model = MVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
    )

    state_dict = torch.load(
        "models/experimental/pointpillars/inputs_weights/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
    )["state_dict"]
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    batch_inputs_dict = torch.load(
        "models/experimental/pointpillars/inputs_weights/batch_inputs_dict_orig.pth", weights_only=False
    )
    batch_data_samples_modified = torch.load(
        "models/experimental/pointpillars/inputs_weights/batch_data_samples_orig.pth", weights_only=False
    )

    output = reference_model(batch_inputs_dict, batch_data_samples_modified)

    for i, out_list in enumerate(output):
        for j, tensor in enumerate(out_list):
            orig = torch.load(f"/home/ubuntu/harini_pointpillars/mmdetection3d/outs_{i}_{j}.pt")
            print(comp_pcc(orig, tensor, 1.0))
