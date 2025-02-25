# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.yolov4.reference.yolov4 import Yolov4
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from models.demos.yolov4.ttnn.yolov4 import TtYOLOv4
from models.demos.yolov4.demo.demo import YoloLayer, get_region_boxes, gen_yolov4_boxes_confs
from models.demos.yolov4.ttnn.weight_parameter_update import update_weight_parameters
from collections import OrderedDict

import cv2
import numpy as np

import pytest
import os


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_yolov4(device, reset_seeds, model_location_generator, use_pretrained_weight):
    torch.manual_seed(0)
    model_path = model_location_generator("models", model_subdir="Yolo")

    if use_pretrained_weight:
        if model_path == "models":
            if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
                os.system(
                    "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
                )  # execute the yolov4_weights_download.sh file

            weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
        else:
            weights_pth = str(model_path / "yolov4.pth")

        ttnn_model = TtYOLOv4(weights_pth, device)
        torch_model = Yolov4()
        new_state_dict = dict(zip(torch_model.state_dict().keys(), ttnn_model.torch_model.values()))
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
    else:
        torch_model = Yolov4.from_random_weights()
        ttnn_weights = update_weight_parameters(OrderedDict(torch_model.state_dict()))
        ttnn_model = TtYOLOv4(ttnn_weights, device)

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

    torch_output_tensor = torch_model(torch_input)

    ref1, ref2, ref3 = gen_yolov4_boxes_confs(torch_output_tensor)
    ref_boxes, ref_confs = get_region_boxes([ref1, ref2, ref3])

    ttnn_output_tensor = ttnn_model(ttnn_input)
    result_boxes_padded = ttnn.to_torch(ttnn_output_tensor[0])
    result_confs = ttnn.to_torch(ttnn_output_tensor[1])

    result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)
    result_boxes_list = []
    # Unpadding
    # That ttnn tensor is the concat output of 3 padded tensors
    # As a perf workaround I'm doing the unpadding on the torch output here.
    # TODO: cleaner ttnn code when ttnn.untilize() is fully optimized
    box_1_start_i = 0
    box_1_end_i = 6100
    box_2_start_i = 6128
    box_2_end_i = 6228
    box_3_start_i = 6256
    box_3_end_i = 6356
    result_boxes_list.append(result_boxes_padded[:, box_1_start_i:box_1_end_i])
    result_boxes_list.append(result_boxes_padded[:, box_2_start_i:box_2_end_i])
    result_boxes_list.append(result_boxes_padded[:, box_3_start_i:box_3_end_i])
    result_boxes = torch.cat(result_boxes_list, dim=1)

    assert_with_pcc(ref_boxes, result_boxes, 0.99)
    assert_with_pcc(ref_confs, result_confs, 0.71)
