# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov4.ttnn.genboxes import TtGenBoxes
from models.experimental.yolov4.demo.demo import YoloLayer
from models.experimental.yolov4.demo.demo import (
    YoloLayer,
    get_region_boxes,
    post_processing,
    plot_boxes_cv2,
    load_class_names,
)

import pytest
import os


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4_post_processing(device, reset_seeds, model_location_generator):
    torch.manual_seed(0)

    torch_input_1 = torch.randn((1, 1, 6400, 256), dtype=torch.bfloat16)
    ttnn_input_1 = ttnn.from_torch(
        torch_input_1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    torch_input_2 = torch.randn((1, 1, 1600, 256), dtype=torch.bfloat16)
    ttnn_input_2 = ttnn.from_torch(
        torch_input_2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    torch_input_3 = torch.randn((1, 1, 400, 256), dtype=torch.bfloat16)
    ttnn_input_3 = ttnn.from_torch(
        torch_input_3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    torch_input_1 = torch_input_1[:, :, :, :255]
    torch_input_1 = torch_input_1.reshape(1, 80, 80, 255)
    torch_input_1 = torch.permute(torch_input_1, (0, 3, 1, 2))
    torch_input_2 = torch_input_2[:, :, :, :255]
    torch_input_2 = torch_input_2.reshape(1, 40, 40, 255)
    torch_input_2 = torch.permute(torch_input_2, (0, 3, 1, 2))
    torch_input_3 = torch_input_3[:, :, :, :255]
    torch_input_3 = torch_input_3.reshape(1, 20, 20, 255)
    torch_input_3 = torch.permute(torch_input_3, (0, 3, 1, 2))

    n_classes = 80

    yolo1 = YoloLayer(
        anchor_mask=[0, 1, 2],
        num_classes=n_classes,
        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        num_anchors=9,
        stride=8,
    )

    yolo2 = YoloLayer(
        anchor_mask=[3, 4, 5],
        num_classes=n_classes,
        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        num_anchors=9,
        stride=16,
    )

    yolo3 = YoloLayer(
        anchor_mask=[6, 7, 8],
        num_classes=n_classes,
        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        num_anchors=9,
        stride=32,
    )

    ref1 = yolo1(torch_input_1)
    ref2 = yolo2(torch_input_2)
    ref3 = yolo3(torch_input_3)

    boxes_confs_1 = TtGenBoxes(device)
    boxes_confs_2 = TtGenBoxes(device)
    boxes_confs_3 = TtGenBoxes(device)

    result_1 = boxes_confs_1(device, ttnn_input_1)
    result_2 = boxes_confs_2(device, ttnn_input_2)
    result_3 = boxes_confs_3(device, ttnn_input_3)

    result_1_bb = ttnn.to_torch(result_1[0])
    result_2_bb = ttnn.to_torch(result_2[0])
    result_3_bb = ttnn.to_torch(result_3[0])

    # print(ref1[0].shape)
    print(result_1_bb.shape, result_2_bb.shape, result_3_bb.shape)

    result_1_bb = result_1_bb.permute(0, 2, 3, 1)
    result_2_bb = result_2_bb.permute(0, 2, 3, 1)
    result_3_bb = result_3_bb.permute(0, 2, 3, 1)

    result_1_bb = result_1_bb.reshape(1, 19200, 1, 4)
    result_2_bb = result_2_bb.reshape(1, 4800, 1, 4)
    result_3_bb = result_3_bb.reshape(1, 1200, 1, 4)

    result_1_conf = ttnn.to_torch(result_1[1])
    result_2_conf = ttnn.to_torch(result_2[1])
    result_3_conf = ttnn.to_torch(result_3[1])

    print("---")
    print(ref1[1].shape)
    print(result_1_conf.shape)

    # result_1_conf = result_1_conf.permute(0, 1, 3, 2)
    # result_2_conf = result_2_conf.permute(0, 1, 3, 2)
    # result_3_conf= result_3_conf.permute(0, 1, 3, 2)

    # result_1_conf = result_1_conf.reshape(1, 4800, 80)
    # result_2_conf = result_2_conf.reshape(1, 1200, 80)
    # result_3_conf= result_3_conf.reshape(1, 300, 80)

    assert_with_pcc(ref1[0], result_1_bb, 0.99)
    assert_with_pcc(ref2[0], result_2_bb, 0.99)
    assert_with_pcc(ref3[0], result_3_bb, 0.99)

    assert_with_pcc(ref1[1], result_1_conf, 0.99)
    assert_with_pcc(ref2[1], result_2_conf, 0.99)
    assert_with_pcc(ref3[1], result_3_conf, 0.99)

    output = get_region_boxes(
        [(result_1_bb, result_1_conf), (result_2_bb, result_2_conf), (result_3_bb, result_3_conf)]
    )
