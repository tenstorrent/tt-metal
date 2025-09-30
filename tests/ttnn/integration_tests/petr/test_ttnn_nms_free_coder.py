# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# ##In Progresss

import torch
import ttnn
import pytest
from models.experimental.functional_petr.tt.ttnn_nms_free_coder import ttnn_NMSFreeCoder
from models.experimental.functional_petr.reference.nms_free_coder import NMSFreeCoder


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_nms_free_coder(device):
    input_torch = torch.load("models/experimental/functional_petr/tt/torch_preds_dicts_nmscoder.pt")
    input_torch_1 = torch.load("models/experimental/functional_petr/tt/torch_preds_dicts_nmscoder.pt")

    input_torch["all_cls_scores"] = ttnn.from_torch(input_torch["all_cls_scores"], device=device)
    input_torch["all_bbox_preds"] = ttnn.from_torch(input_torch["all_bbox_preds"], device=device)

    torch_model = NMSFreeCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # self.point_cloud_range,
        max_num=300,
        voxel_size=[0.2, 0.2, 8],  # self.voxel_size,
        num_classes=10,
    )

    ttnn_model = ttnn_NMSFreeCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # self.point_cloud_range,
        max_num=300,
        voxel_size=[0.2, 0.2, 8],  # self.voxel_size,
        num_classes=10,
    )

    torch_output = torch_model.decode(input_torch_1)

    output = ttnn_model.decode(input_torch)

    print("torch_output shapes")
    for i in torch_output[0].keys():
        print(torch_output[0][i].shape)

    print("ttnn_output shapes")
    for i in output[0].keys():
        print(output[0][i].shape)
