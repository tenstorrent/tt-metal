# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.yolov4.reference.yolov4 import Yolov4
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
from models.demos.yolov4.ttnn.yolov4 import TtYOLOv4
from models.demos.yolov4.demo.demo import YoloLayer, get_region_boxes, post_processing, plot_boxes_cv2, load_class_names
import cv2
import numpy as np

import pytest
import os


def gen_yolov4_boxes_confs(output):
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

    y1 = yolo1(output[0])
    y2 = yolo2(output[1])
    y3 = yolo3(output[2])

    return y1, y2, y3


@skip_for_grayskull()
@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4(device, reset_seeds, model_location_generator):
    torch.manual_seed(0)
    model_path = model_location_generator("models", model_subdir="Yolo")

    if model_path == "models":
        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file

        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")

    ttnn_model = TtYOLOv4(weights_pth, device)

    imgfile = "models/demos/yolov4/demo/giraffe_320.jpg"
    width = 320
    height = 320
    img = cv2.imread(imgfile)
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    torch_input = torch.autograd.Variable(img)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16)

    torch_model = Yolov4()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items()}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_output_tensor = torch_model(torch_input)

    ref1, ref2, ref3 = gen_yolov4_boxes_confs(torch_output_tensor)
    ref_boxes, ref_confs = get_region_boxes([ref1, ref2, ref3])

    ttnn_output_tensor = ttnn_model(device, ttnn_input)
    result_boxes_padded = ttnn.to_torch(ttnn_output_tensor[0])
    result_confs = ttnn.to_torch(ttnn_output_tensor[1])

    result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)
    result_boxes_list = []
    result_boxes_list.append(result_boxes_padded[:, 0:6100])
    result_boxes_list.append(result_boxes_padded[:, 6128:6228])
    result_boxes_list.append(result_boxes_padded[:, 6256:6356])
    result_boxes = torch.cat(result_boxes_list, dim=1)

    print(ref_boxes.shape, ref_confs.shape)
    print(result_boxes.shape, result_confs.shape)

    assert_with_pcc(ref_boxes, result_boxes, 0.99)
    assert_with_pcc(ref_confs, result_confs, 0.71)

    """
    ttnn_output_tensor = ttnn_model(device, ttnn_input)
    result_1 = ttnn_output_tensor[0]
    result_2 = ttnn_output_tensor[1]
    result_3 = ttnn_output_tensor[2]

    result_1_bb = ttnn.to_torch(result_1[0])
    result_2_bb = ttnn.to_torch(result_2[0])
    result_3_bb = ttnn.to_torch(result_3[0])

    result_1_bb = result_1_bb.permute(0, 2, 3, 1)
    result_2_bb = result_2_bb.permute(0, 2, 3, 1)
    result_3_bb = result_3_bb.permute(0, 2, 3, 1)

    result_1_bb = result_1_bb.reshape(1, 4800, 1, 4)
    result_2_bb = result_2_bb.reshape(1, 1200, 1, 4)
    result_3_bb = result_3_bb.reshape(1, 300, 1, 4)

    result_1_conf = ttnn.to_torch(result_1[1])
    result_2_conf = ttnn.to_torch(result_2[1])
    result_3_conf = ttnn.to_torch(result_3[1])

    result_1_conf = result_1_conf.permute(0, 2, 1)
    result_2_conf = result_2_conf.permute(0, 2, 1)
    result_3_conf = result_3_conf.permute(0, 2, 1)

    assert_with_pcc(ref1[0], result_1_bb, 0.99)
    assert_with_pcc(ref2[0], result_2_bb, 0.99)
    assert_with_pcc(ref3[0], result_3_bb, 0.99)

    # assert_with_pcc(ref1[1], result_1_conf, 0.99)
    # assert_with_pcc(ref2[1], result_2_conf, 0.99)
    # assert_with_pcc(ref3[1], result_3_conf, 0.99)

    output = get_region_boxes(
        [(result_1_bb, result_1_conf), (result_2_bb, result_2_conf), (result_3_bb, result_3_conf)]
    )
    """

    ## Giraffe image detection
    conf_thresh = 0.3
    nms_thresh = 0.4
    output = [result_boxes.to(torch.float16), result_confs.to(torch.float16)]

    boxes = post_processing(img, conf_thresh, nms_thresh, output)
    namesfile = "models/demos/yolov4/demo/coco.names"
    class_names = load_class_names(namesfile)
    img = cv2.imread(imgfile)
    plot_boxes_cv2(img, boxes[0], "ttnn_prediction_demo.jpg", class_names)
