# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest
import torch

import ttnn
from models.demos.yolov4.common import (
    YOLOV4_BOXES_PCC,
    YOLOV4_BOXES_PCC_BLACKHOLE,
    YOLOV4_CONFS_PCC,
    image_to_tensor,
    load_image,
    load_torch_model,
)
from models.demos.yolov4.post_processing import gen_yolov4_boxes_confs, get_region_boxes
from models.demos.yolov4.reference.yolov4 import Yolov4
from models.demos.yolov4.runner.runner import YOLOv4Runner
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters
from models.demos.yolov4.tt.weight_parameter_update import update_weight_parameters
from models.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_yolov4(device, reset_seeds, model_location_generator, use_pretrained_weight, resolution):
    torch.manual_seed(0)

    if use_pretrained_weight:
        torch_model = load_torch_model(model_location_generator)
    else:
        torch_model = Yolov4.from_random_weights()
        ttnn_weights = update_weight_parameters(OrderedDict(torch_model.state_dict()))
        torch_dict = ttnn_weights
        new_state_dict = dict(zip(torch_model.state_dict().keys(), torch_dict.values()))
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()

    imgfile = "models/demos/yolov4/resources/giraffe_320.jpg"
    img = load_image(imgfile, resolution)
    torch_input = image_to_tensor(img)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16)

    torch_output_tensor = torch_model(torch_input)

    parameters = create_yolov4_model_parameters(torch_model, torch_input, resolution, device)

    ref1, ref2, ref3 = gen_yolov4_boxes_confs(torch_output_tensor)
    ref_boxes, ref_confs = get_region_boxes([ref1, ref2, ref3])

    ttnn_model_runner = YOLOv4Runner(device, parameters, resolution)
    result_boxes, result_confs = ttnn_model_runner.run(ttnn_input)

    if is_blackhole():
        assert_with_pcc(ref_boxes, result_boxes, YOLOV4_BOXES_PCC_BLACKHOLE)
    else:
        assert_with_pcc(ref_boxes, result_boxes, YOLOV4_BOXES_PCC)
        assert_with_pcc(ref_confs, result_confs, YOLOV4_CONFS_PCC)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384}],
    indirect=True,
    ids=["0"],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
    ids=[
        "0",
        "1",
    ],
)
def test_yolov4(device, reset_seeds, model_location_generator, use_pretrained_weight, resolution):
    run_yolov4(
        device,
        reset_seeds,
        model_location_generator,
        use_pretrained_weight,
        resolution,
    )
